// =============================================================================
// gemm_c500.cu
//
//   C = alpha * A * B + beta * C        (矩阵乘加 / GEMM)
//   A: M x K  fp16 行主序
//   B: K x N  fp16 行主序 (或 N x K,见 MX_B_COLMAJOR)
//   C: M x N  fp32 行主序
//
//   目标平台: 沐曦 C500 (MACA),编译器 mxcc / cucc
//   Block tile: BM x BN = 128 x 128,K 方向宏 tile = 256
//   线程组织:  256 线程 = 4 wave,wave 排成 2x2,每 wave 算 64x64
//
// 同一份 kernel 源码通过模板参数产生多个版本,便于逐步对比加速比:
//   v0  gemm_naive                      1by1 标量搬运,无共享内存
//   v1  <STAGES=1, SWZ=0, PF=0>         满足 MMA 的搬运布局,单缓冲
//   v2  <STAGES=1, SWZ=1, PF=0>         + swizzle 解 bank conflict
//   v3  <STAGES=3, SWZ=1, PF=1>         + 多 stage + 寄存器预取流水
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "mx_mma.h"

// -----------------------------------------------------------------------------
// 分块参数
// -----------------------------------------------------------------------------
#ifndef BM
#define BM 128   // 一个 block 负责的 C 行数
#endif
#ifndef BN
#define BN 128   // 一个 block 负责的 C 列数
#endif
#ifndef BK
#define BK 32    // 一个 stage 在 K 方向搬运的深度
#endif
#ifndef KTILE
#define KTILE 256  // K 方向宏 tile:断言粒度 / 将来 split-K 的切分单位
#endif

#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

#define WAVES_M 2                  // wave 2x2
#define WAVES_N 2
#define NWAVES  (WAVES_M * WAVES_N)
#define THREADS (NWAVES * MX_WAVE)  // 4 * 64 = 256

#define WM (BM / WAVES_M)          // 每 wave 的 M 方向 tile = 64
#define WN (BN / WAVES_N)          // 每 wave 的 N 方向 tile = 64
#define TM (WM / MMA_M)            // 每 wave 在 M 方向的 MMA tile 数 = 4
#define TN (WN / MMA_N)            // 每 wave 在 N 方向的 MMA tile 数 = 4

#define KSTEPS_PER_STAGE (BK / MMA_K)   // 一个 stage 里走几次 MMA 的 K = 2

// 每线程每 stage 的搬运份数(以 8B = 4×fp16 的 chunk 为单位)
#define A_CHUNKS_PER_THREAD ((BM * BK / 4) / THREADS)   // 128*32/4/256 = 4
#define B_CHUNKS_PER_THREAD ((BN * BK / 4) / THREADS)   // 同上 = 4

#ifndef MX_B_COLMAJOR
#define MX_B_COLMAJOR 0   // 1 表示 B 以 N x K 列主序(= B^T 行主序)给进来
#endif

// -----------------------------------------------------------------------------
// 共享内存索引 + swizzle
//
// 逻辑布局都是 [row][k],k 连续 —— 因为 MMA 的 A/B 片段要的正好是"固定 row、
// 4 个连续 k",k 连续才能用一条 8B 的 LDS 读指令取完一个片段。
//
// 但裸 [row][BK] 布局会 bank conflict:行跨距 = BK*2 = 64B = 16 dword,
// 对 32 bank 取模只有 {0,16} 两个值,一个 MMA 片段要 16 个 lane 各读 8B,
// 全挤在 2 组 bank 上 → 8 路冲突。
//
// swizzle 做法:把 k 方向切成 8B 的 chunk(4 个 fp16 一个 chunk,BK=32 → 8 chunk),
// 用行号去 XOR chunk 号:
//        chunk' = chunk ^ ((row >> 1) & (BK/4 - 1))
// 为什么是 (row>>1):行跨距对 bank 取模只贡献行号的最低 1 位(0 或 16),
// 所以剩下的 3 位必须由 XOR 来打散,正好 2 × 8 = 16,16 个 lane 各占 2 个 dword,
// 刚好铺满 32 个 bank 各一次 —— 读侧完全无冲突。
// 写侧(global->bsm)同一个公式算下来也是满铺无冲突,推导见 README。
// -----------------------------------------------------------------------------
#define SWZ_MASK (BK / 4 - 1)

template <int SWZ>
__device__ __forceinline__ int sidx(int row, int k) {
  int chunk = k >> 2;          // 8B chunk 号
  int off   = k & 3;           // chunk 内偏移
  if (SWZ) chunk ^= ((row >> 1) & SWZ_MASK);
  return row * BK + chunk * 4 + off;
}

// =============================================================================
// v0: 最朴素版本 —— 数据 1by1 搬,每个线程算一个 C 元素
//
// 作用:(1) 功能基线,用来和优化版对数;(2) 让"访存量"这件事变得直观 ——
// 这个版本每算一次乘加就要从 global 读 A、B 各一个 fp16,
// 算术强度 = 2 FLOP / 4 Byte = 0.5 FLOP/Byte,被带宽彻底锁死。
// =============================================================================
__global__ void gemm_naive(const mx_h *__restrict__ A, const mx_h *__restrict__ B,
                           float *__restrict__ C, int M, int N, int K,
                           float alpha, float beta) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
#if MX_B_COLMAJOR
    acc += (float)A[(size_t)m * K + k] * (float)B[(size_t)n * K + k];
#else
    acc += (float)A[(size_t)m * K + k] * (float)B[(size_t)k * N + n];
#endif
  }
  C[(size_t)m * N + n] = alpha * acc + beta * C[(size_t)m * N + n];
}

#if MX_HAS_MMA

// =============================================================================
// v1/v2/v3: MMA 版本
//
// 模板参数
//   STAGES : 共享内存缓冲数。1 = 单缓冲(搬完必须等);>=2 = 多 stage 流水
//   SWZ    : 1 = 开 swizzle 解 bank conflict
//   PF     : 1 = 预取到寄存器再落 bsm(或用异步拷贝),把 global 延迟藏进计算里
// =============================================================================
template <int STAGES, int SWZ, int PF>
__global__ __launch_bounds__(THREADS) void gemm_mma(
    const mx_h *__restrict__ A, const mx_h *__restrict__ B,
    float *__restrict__ C, int M, int N, int K, float alpha, float beta) {

  // ---------------------------------------------------------------------------
  // 共享内存:A 和 B 各 STAGES 份。
  // 单 stage = (128*32 + 128*32) * 2B = 16 KB,STAGES=3 → 48 KB,STAGES=4 → 64 KB
  // ---------------------------------------------------------------------------
  __shared__ mx_h sA[STAGES][BM * BK];
  __shared__ mx_h sB[STAGES][BN * BK];

  const int tid  = threadIdx.x;
  const int wave = tid / MX_WAVE;        // 0..3
  const int lane = tid % MX_WAVE;        // 0..63

  // wave 2x2:wave 在 block 内的二维坐标 -> 它负责的 64x64 子块原点
  const int wave_m = (wave / WAVES_N) * WM;
  const int wave_n = (wave % WAVES_N) * WN;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  // ---------------------------------------------------------------------------
  // 搬运索引预计算(循环里只做加法,不重复做除法/取模)
  //
  // A 的搬运:tile 是 BM 行 × BK 列,行内 BK*2 = 64B = 4 个 16B。
  //   把 tile 看成 (BM*BK/8) 个 16B 单元,256 线程每人 2 个。
  //   连续 tid 落在同一行的相邻 16B 上 → 每 4 个线程凑出 64B 连续访存。
  // ---------------------------------------------------------------------------
  const int a_vec_per_row = BK / 8;                      // 每行 4 个 16B
  const int a_units       = BM * BK / 8;                 // 512
  const int a_per_thread  = a_units / THREADS;           // 2
  const int a_row0        = (tid / a_vec_per_row);        // 本线程第 0 份的行
  const int a_k0          = (tid % a_vec_per_row) * 8;    // 本线程的 k 起点
  const int a_row_stride  = THREADS / a_vec_per_row;      // 每份之间跨多少行 = 64

#if MX_B_COLMAJOR
  // B 已是 N x K 行主序,和 A 完全对称
  const int b_row0       = (tid / a_vec_per_row);
  const int b_k0         = (tid % a_vec_per_row) * 8;
  const int b_row_stride = THREADS / a_vec_per_row;
#else
  // B 是 K x N 行主序,但共享内存要 [n][k] —— 必须在搬运时转置。
  //   取法:一个线程包一列 n 的一段 k。
  //   n = tid % BN  → 同一 wave 内连续 lane 读连续的 n,
  //                   一条指令 64 lane × 2B = 128B 连续,global 侧仍然合并。
  //   k 段 = (tid / BN) * (BK / (THREADS/BN)) 开始的 16 个 k。
  // 代价:global 侧每条指令只搬 128B(A 侧是 1KB),指令数多 8 倍。
  //      这是 B 行主序的固有代价,想消掉就把 B 预转置成 N x K(-DMX_B_COLMAJOR=1),
  //      转置是 O(KN),相对 O(MNK) 可忽略。
  const int b_n        = tid % BN;
  const int b_k_group  = tid / BN;                        // 0..1
  const int b_k_per_th = BK / (THREADS / BN);             // 32/2 = 16
  const int b_k_base   = b_k_group * b_k_per_th;
#endif

  // ---------------------------------------------------------------------------
  // 预取寄存器(PF=1 时用)
  // A: 2 份 × 16B = 8 个 VGPR;B: 4 个 8B chunk = 8 个 VGPR
  // ---------------------------------------------------------------------------
  mx_h8 ra[a_per_thread > 0 ? a_per_thread : 1];
  mx_h4 rb[B_CHUNKS_PER_THREAD];

  // ---- global -> 寄存器 ----
  auto g2r = [&](int k0) {
#pragma unroll
    for (int i = 0; i < a_per_thread; ++i) {
      int row = a_row0 + i * a_row_stride;
      ra[i] = *(const mx_h8 *)(A + (size_t)(block_m + row) * K + k0 + a_k0);
    }
#if MX_B_COLMAJOR
#pragma unroll
    for (int i = 0; i < a_per_thread; ++i) {
      int row = b_row0 + i * b_row_stride;
      *((mx_h8 *)rb + i) = *(const mx_h8 *)(B + (size_t)(block_n + row) * K + k0 + b_k0);
    }
#else
#pragma unroll
    for (int c = 0; c < B_CHUNKS_PER_THREAD; ++c) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int k = b_k_base + c * 4 + j;
        rb[c][j] = B[(size_t)(k0 + k) * N + block_n + b_n];
      }
    }
#endif
  };

  // ---- 寄存器 -> bsm ----
  auto r2s = [&](int buf) {
#pragma unroll
    for (int i = 0; i < a_per_thread; ++i) {
      int row = a_row0 + i * a_row_stride;
      // 一个 16B 单元 = 2 个 8B chunk,swizzle 粒度是 8B,所以拆成两条 8B 写
      mx_h4 lo, hi;
#pragma unroll
      for (int j = 0; j < 4; ++j) { lo[j] = ra[i][j]; hi[j] = ra[i][j + 4]; }
      mx_lds_write8(&sA[buf][sidx<SWZ>(row, a_k0)],     lo);
      mx_lds_write8(&sA[buf][sidx<SWZ>(row, a_k0 + 4)], hi);
    }
#if MX_B_COLMAJOR
#pragma unroll
    for (int i = 0; i < a_per_thread; ++i) {
      int row = b_row0 + i * b_row_stride;
      const mx_h8 v = *((const mx_h8 *)rb + i);
      mx_h4 lo, hi;
#pragma unroll
      for (int j = 0; j < 4; ++j) { lo[j] = v[j]; hi[j] = v[j + 4]; }
      mx_lds_write8(&sB[buf][sidx<SWZ>(row, b_k0)],     lo);
      mx_lds_write8(&sB[buf][sidx<SWZ>(row, b_k0 + 4)], hi);
    }
#else
#pragma unroll
    for (int c = 0; c < B_CHUNKS_PER_THREAD; ++c)
      mx_lds_write8(&sB[buf][sidx<SWZ>(b_n, b_k_base + c * 4)], rb[c]);
#endif
  };

  // ---- 不走寄存器的直通搬运(PF=0,v1/v2 用) ----
  auto g2s = [&](int k0, int buf) { g2r(k0); r2s(buf); };

  // ---------------------------------------------------------------------------
  // 累加器:每 wave 4x4 个 MMA tile,每 tile 每 lane 4 个 fp32
  //   => 16 * 4 = 64 个 fp32 寄存器/lane。这是本配置的核心预算。
  //   (若 block tile 放大到 256x256,这里会变成 256 个,极易 spill)
  // ---------------------------------------------------------------------------
  mx_f4 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (mx_f4){0.f, 0.f, 0.f, 0.f};

  // MMA 片段在 tile 内的 lane 偏移
  const int frag_row = mx_frag_a_row(lane);   // = lane % 16
  const int frag_col = mx_frag_b_col(lane);   // = lane % 16
  const int frag_k   = mx_frag_k(lane);       // = (lane/16) * 4

  // ---- 一个 stage 的计算:BK/16 = 2 次 MMA-K,每次 4x4 = 16 条 MMA ----
  auto compute = [&](int buf) {
#pragma unroll
    for (int ks = 0; ks < KSTEPS_PER_STAGE; ++ks) {
      const int kbase = ks * MMA_K + frag_k;

      // 先把 4 个 A 片段和 4 个 B 片段读到寄存器,再做 16 条 MMA。
      // 这就是 wave 级的 2x2(这里是 4x4)寄存器分块:
      //   8 次 LDS 读 换 16 条 MMA → 每个片段被复用 4 次,
      //   LDS 带宽压力直接降到 1/4。不做分块的话要 32 次读。
      mx_h4 fa[TM], fb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        fa[i] = mx_lds_read8(&sA[buf][sidx<SWZ>(wave_m + i * MMA_M + frag_row, kbase)]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        fb[j] = mx_lds_read8(&sB[buf][sidx<SWZ>(wave_n + j * MMA_N + frag_col, kbase)]);

#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = mx_mma_16x16x16(fa[i], fb[j], acc[i][j]);
    }
  };

  // ---------------------------------------------------------------------------
  // 主循环 / 流水
  //
  // nk = K / BK 个 stage 步。K 必须是 KTILE(256) 的整数倍,见 host 侧断言。
  //
  // STAGES = 1(v1/v2):搬 -> 同步 -> 算 -> 同步,global 延迟完全暴露。
  // STAGES >= 2(v3): 开场先灌满 STAGES-1 个缓冲;循环里"先发下一批 global 读,
  //   再算当前 stage" —— global 的几百拍延迟被 32 条 MMA 盖住。
  //
  // 为什么一次迭代只要一个 __syncthreads():
  //   第 kt 轮写入的缓冲是 (kt+STAGES-1) % STAGES = (kt-1) % STAGES,
  //   也就是第 kt-1 轮读过的那个;而第 kt-1 轮末尾的 barrier 已经保证
  //   所有线程都算完了 kt-1。所以写它不会撞上任何还在读的人。
  // ---------------------------------------------------------------------------
  const int nk = K / BK;

  if (STAGES == 1) {
    for (int kt = 0; kt < nk; ++kt) {
      g2s(kt * BK, 0);
      __syncthreads();
      compute(0);
      __syncthreads();
    }
  } else {
    // 预热:灌入 STAGES-1 个 stage
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < nk) g2s(s * BK, s);
    }
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
      const int cur  = kt % STAGES;
      const int nk_i = kt + STAGES - 1;          // 本轮要预取的 stage 序号

      if (PF && nk_i < nk) g2r(nk_i * BK);       // 先发射 global 读
      compute(cur);                              // 用计算盖住它的延迟
      if (nk_i < nk) {
        if (PF) r2s(nk_i % STAGES);
        else    g2s(nk_i * BK, nk_i % STAGES);
      }
      __syncthreads();
    }
  }

  // ---------------------------------------------------------------------------
  // store C
  //   布局 0 下,同一 tile 内 16 个 lane 的 n 连续 → 16 × 4B = 64B 合并写。
  //   4 个元素在 M 方向跨 1 行,所以是 4 条独立的 4B 写(不能向量化)。
  //   若 MX_FRAG_LAYOUT=1(n 方向连续),这里可以换成一条 16B 写。
  // ---------------------------------------------------------------------------
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = block_m + wave_m + i * MMA_M + mx_frag_c_m(lane, e);
        const int n = block_n + wave_n + j * MMA_N + mx_frag_c_n(lane, e);
        float *p = C + (size_t)m * N + n;
        *p = alpha * acc[i][j][e] + beta * (*p);
      }
    }
  }
}

// =============================================================================
// MMA 片段布局自检探针
//
// 用一块 16x16x16 验证 mx_mma.h 里假设的 lane->元素映射对不对。
// 不对的话换 -DMX_FRAG_LAYOUT=1 再试;两个都不对就把实测的 C 打出来反推。
// =============================================================================
__global__ void mma_probe(const mx_h *A16, const mx_h *B16, float *C16) {
  const int lane = threadIdx.x;
  mx_h4 a, b;
  mx_f4 c = {0.f, 0.f, 0.f, 0.f};

  const int m  = mx_frag_a_row(lane);
  const int n  = mx_frag_b_col(lane);
  const int k0 = mx_frag_k(lane);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[i] = A16[m * 16 + k0 + i];        // A[m][k]
    b[i] = B16[(k0 + i) * 16 + n];      // B[k][n]
  }
  c = mx_mma_16x16x16(a, b, c);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    C16[mx_frag_c_m(lane, i) * 16 + mx_frag_c_n(lane, i)] = c[i];
}

#endif  // MX_HAS_MMA

// =============================================================================
// 以下为 host 侧:数据准备、正确性校验、计时与利用率
// =============================================================================

static void cpu_gemm(const mx_h_store *A, const mx_h_store *B, float *C, int M,
                     int N, int K, float alpha, float beta) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
#if MX_B_COLMAJOR
        acc += mx_h2f(A[(size_t)m * K + k]) * mx_h2f(B[(size_t)n * K + k]);
#else
        acc += mx_h2f(A[(size_t)m * K + k]) * mx_h2f(B[(size_t)k * N + n]);
#endif
      }
      C[(size_t)m * N + n] = alpha * acc + beta * C[(size_t)m * N + n];
    }
}

struct Result { double ms; double tflops; bool ok; double max_rel; };

// 相对误差:fp16 输入 + fp32 累加,K=4096 时 1e-2 的相对误差是正常的
static void compare(const float *ref, const float *got, size_t n, double *max_rel,
                    bool *ok, double tol) {
  double mr = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)ref[i] - (double)got[i]);
    double r = d / (fabs((double)ref[i]) + 1e-6);
    if (r > mr) mr = r;
  }
  *max_rel = mr;
  *ok = (mr < tol);
}

int main(int argc, char **argv) {
  int M = 4096, N = 4096, K = 4096;
  int iters = 50, warmup = 10;
  double peak_tflops = 0.0;   // --peak 给进来才能算利用率
  double peak_bw_gbs = 0.0;
  bool do_probe = false, do_check = true;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) M = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-n") && i + 1 < argc) N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-k") && i + 1 < argc) K = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--peak") && i + 1 < argc) peak_tflops = atof(argv[++i]);
    else if (!strcmp(argv[i], "--bw") && i + 1 < argc) peak_bw_gbs = atof(argv[++i]);
    else if (!strcmp(argv[i], "--probe")) do_probe = true;
    else if (!strcmp(argv[i], "--no-check")) do_check = false;
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      printf("用法: %s [-m M] [-n N] [-k K] [--iters I] [--peak TFLOPS] "
             "[--bw GB/s] [--probe] [--no-check]\n", argv[0]);
      return 0;
    }
  }

  printf("================ 配置 ================\n");
  printf("问题规模       : M=%d N=%d K=%d\n", M, N, K);
  printf("block tile     : %d x %d, K 宏 tile = %d, stage 深度 BK = %d\n",
         BM, BN, KTILE, BK);
  printf("线程组织       : %d 线程 = %d wave (%dx%d), 每 wave %dx%d, "
         "每 wave %dx%d 个 %dx%dx%d MMA\n",
         THREADS, NWAVES, WAVES_M, WAVES_N, WM, WN, TM, TN, MMA_M, MMA_N, MMA_K);
  printf("累加器         : %d 个 fp32/lane\n", TM * TN * 4);
  printf("共享内存/stage : %.1f KB  (A %.1f + B %.1f)\n",
         (BM * BK + BN * BK) * 2 / 1024.0, BM * BK * 2 / 1024.0,
         BN * BK * 2 / 1024.0);
  printf("MX_HAS_MMA=%d  MX_HAS_BSM_ASYNC=%d  MX_FRAG_LAYOUT=%d  "
         "MX_B_COLMAJOR=%d  MX_WAVE=%d\n",
         MX_HAS_MMA, MX_HAS_BSM_ASYNC, MX_FRAG_LAYOUT, MX_B_COLMAJOR, MX_WAVE);
  printf("======================================\n\n");

  if (M % BM || N % BN || K % KTILE) {
    fprintf(stderr, "错误: 要求 M%%%d==0, N%%%d==0, K%%%d==0\n", BM, BN, KTILE);
    return 1;
  }

#if !MX_HAS_MMA
  printf("!! MX_HAS_MMA=0:MMA kernel 未参与编译,只会跑 v0 基线。\n");
  printf("   在真机上按 mx_mma.h 顶部的 grep 命令确认 builtin 名字后,\n");
  printf("   用 make HAS_MMA=1 重新编译。\n\n");
#endif

  // ---------------- 片段布局自检 ----------------
#if MX_HAS_MMA
  if (do_probe) {
    const int T = 16;
    std::vector<mx_h_store> hA(T * T), hB(T * T);
    std::vector<float> hC(T * T, 0.f), hRef(T * T, 0.f);
    for (int i = 0; i < T * T; ++i) {
      hA[i] = mx_f2h((float)((i * 7 % 13) - 6) * 0.25f);
      hB[i] = mx_f2h((float)((i * 5 % 11) - 5) * 0.5f);
    }
    for (int m = 0; m < T; ++m)
      for (int n = 0; n < T; ++n) {
        float a = 0;
        for (int k = 0; k < T; ++k) a += mx_h2f(hA[m * T + k]) * mx_h2f(hB[k * T + n]);
        hRef[m * T + n] = a;
      }
    mx_h *dA, *dB; float *dC;
    MX_CHECK(cudaMalloc(&dA, T * T * 2));
    MX_CHECK(cudaMalloc(&dB, T * T * 2));
    MX_CHECK(cudaMalloc(&dC, T * T * 4));
    MX_CHECK(cudaMemcpy(dA, hA.data(), T * T * 2, cudaMemcpyHostToDevice));
    MX_CHECK(cudaMemcpy(dB, hB.data(), T * T * 2, cudaMemcpyHostToDevice));
    MX_CHECK(cudaMemset(dC, 0, T * T * 4));
    mma_probe<<<1, MX_WAVE>>>(dA, dB, dC);
    MX_CHECK(cudaDeviceSynchronize());
    MX_CHECK(cudaMemcpy(hC.data(), dC, T * T * 4, cudaMemcpyDeviceToHost));
    double mr; bool ok;
    compare(hRef.data(), hC.data(), T * T, &mr, &ok, 1e-2);
    printf("[probe] MX_FRAG_LAYOUT=%d -> %s (max rel err = %.3e)\n",
           MX_FRAG_LAYOUT, ok ? "布局假设正确" : "布局假设错误", mr);
    if (!ok) {
      printf("        换 -DMX_FRAG_LAYOUT=1 重试;若仍不对,下面是实测 vs 期望的\n"
             "        第 0 行,用来反推真实映射:\n        got :");
      for (int j = 0; j < T; ++j) printf(" %6.1f", hC[j]);
      printf("\n        want:");
      for (int j = 0; j < T; ++j) printf(" %6.1f", hRef[j]);
      printf("\n");
    }
    printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (!ok) return 1;
  }
#endif

  // ---------------- 数据准备 ----------------
  const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
  std::vector<mx_h_store> hA(szA), hB(szB);
  std::vector<float> hC0(szC), hC(szC), hRef(szC);

  srand(1234);
  for (size_t i = 0; i < szA; ++i) hA[i] = mx_f2h((float)(rand() % 17 - 8) * 0.125f);
  for (size_t i = 0; i < szB; ++i) hB[i] = mx_f2h((float)(rand() % 17 - 8) * 0.125f);
  for (size_t i = 0; i < szC; ++i) hC0[i] = (float)(rand() % 5 - 2);

  const float alpha = 1.0f, beta = 0.5f;

  mx_h *dA, *dB; float *dC;
  MX_CHECK(cudaMalloc(&dA, szA * 2));
  MX_CHECK(cudaMalloc(&dB, szB * 2));
  MX_CHECK(cudaMalloc(&dC, szC * 4));
  MX_CHECK(cudaMemcpy(dA, hA.data(), szA * 2, cudaMemcpyHostToDevice));
  MX_CHECK(cudaMemcpy(dB, hB.data(), szB * 2, cudaMemcpyHostToDevice));

  // 参考值:小规模用 CPU 直接算;大规模用 CPU 太慢,改成和 v0 对数
  bool have_cpu_ref = false;
  if (do_check && (double)M * N * K <= 2.0e9) {
    memcpy(hRef.data(), hC0.data(), szC * 4);
    cpu_gemm(hA.data(), hB.data(), hRef.data(), M, N, K, alpha, beta);
    have_cpu_ref = true;
  }

  cudaEvent_t e0, e1;
  MX_CHECK(cudaEventCreate(&e0));
  MX_CHECK(cudaEventCreate(&e1));

  // 大规模下 CPU 参考太慢,改用 v0 的 GPU 结果当参考(v0 最先跑)
  std::vector<float> gpuRef;
  bool have_gpu_ref = false;

  auto run = [&](const char *name, auto launch) -> Result {
    Result r{0, 0, true, 0};
    // 正确性
    if (do_check) {
      MX_CHECK(cudaMemcpy(dC, hC0.data(), szC * 4, cudaMemcpyHostToDevice));
      launch();
      MX_CHECK(cudaDeviceSynchronize());
      MX_CHECK(cudaGetLastError());
      MX_CHECK(cudaMemcpy(hC.data(), dC, szC * 4, cudaMemcpyDeviceToHost));
      if (have_cpu_ref) {
        compare(hRef.data(), hC.data(), szC, &r.max_rel, &r.ok, 2e-2);
      } else if (!have_gpu_ref) {
        gpuRef = hC;           // v0 的输出,后续版本都和它对
        have_gpu_ref = true;
        r.max_rel = -1;
      } else {
        compare(gpuRef.data(), hC.data(), szC, &r.max_rel, &r.ok, 2e-2);
      }
    }
    // 计时
    for (int i = 0; i < warmup; ++i) launch();
    MX_CHECK(cudaDeviceSynchronize());
    MX_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch();
    MX_CHECK(cudaEventRecord(e1));
    MX_CHECK(cudaEventSynchronize(e1));
    float ms;
    MX_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    r.ms = ms / iters;
    r.tflops = 2.0 * M * N * K / (r.ms * 1e-3) / 1e12;
    printf("%-34s %9.3f ms  %8.2f TFLOPS", name, r.ms, r.tflops);
    if (peak_tflops > 0) printf("  算力利用率 %6.2f%%", r.tflops / peak_tflops * 100);
    if (do_check) {
      if (r.max_rel < 0) printf("  [作为参考基准]");
      else printf("  [%s rel=%.2e]", r.ok ? "PASS" : "FAIL", r.max_rel);
    }
    printf("\n");
    return r;
  };

  printf("---------------- 结果 ----------------\n");

  Result r0 = run("v0 naive (1by1)", [&] {
    dim3 blk(32, 8), grd((N + 31) / 32, (M + 7) / 8);
    gemm_naive<<<grd, blk>>>(dA, dB, dC, M, N, K, alpha, beta);
  });

#if MX_HAS_MMA
  dim3 grd(N / BN, M / BM), blk(THREADS);
  Result r1 = run("v1 MMA + 单缓冲 + 无 swizzle", [&] {
    gemm_mma<1, 0, 0><<<grd, blk>>>(dA, dB, dC, M, N, K, alpha, beta);
  });
  Result r2 = run("v2 + swizzle 解 bank conflict", [&] {
    gemm_mma<1, 1, 0><<<grd, blk>>>(dA, dB, dC, M, N, K, alpha, beta);
  });
  Result r3 = run("v3 + 3 stage + 寄存器预取流水", [&] {
    gemm_mma<3, 1, 1><<<grd, blk>>>(dA, dB, dC, M, N, K, alpha, beta);
  });
  Result r4 = run("v4 + 4 stage", [&] {
    gemm_mma<4, 1, 1><<<grd, blk>>>(dA, dB, dC, M, N, K, alpha, beta);
  });

  printf("\n---------------- 分析 ----------------\n");
  printf("v1 相对 v0 加速       : %.1fx\n", r0.ms / r1.ms);
  printf("swizzle 收益          : %+.1f%%\n", (r1.ms / r2.ms - 1) * 100);
  printf("多 stage + 流水收益   : %+.1f%%\n", (r2.ms / r3.ms - 1) * 100);
  printf("4 stage vs 3 stage    : %+.1f%%\n", (r3.ms / r4.ms - 1) * 100);

  const Result best = r4.ms < r3.ms ? r4 : r3;
  // HBM 最小访存量:A、B 各按 block tile 复用后读一遍,C 读+写
  double traffic = (double)M * K * 2 * (N / BN) + (double)K * N * 2 * (M / BM) +
                   (double)M * N * 4 * 2;
  double bw = traffic / (best.ms * 1e-3) / 1e9;
  printf("最佳版本有效 HBM 带宽 : %.1f GB/s", bw);
  if (peak_bw_gbs > 0) printf("  (带宽利用率 %.1f%%)", bw / peak_bw_gbs * 100);
  printf("\n算术强度              : %.1f FLOP/Byte\n",
         2.0 * M * N * K / traffic);
  if (peak_tflops <= 0)
    printf("\n提示: 传 --peak <C500 的 fp16 峰值 TFLOPS> 才能打印算力利用率,\n"
           "      传 --bw <HBM 峰值 GB/s> 才能打印带宽利用率。\n"
           "      这两个数请从沐曦官方规格书取,我不替你编。\n");
#endif

  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
