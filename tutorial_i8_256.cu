// =============================================================================
// tutorial_i8_256.cu
//   沐曦 C500 (xcore1000) int8 -> int32 MMA 教程算子
//   TileMNK = 256 x 256 x 256
//
// 硬件事实(本机实测 / SDK 原文,见教程正文):
//   arch        xcore1000      -> int8 MMA 只有 16x16x16(32 要 __MACA_ARCH__==1500)
//   warpSize    64             -> 一个 wave 64 lane
//   LDS         65536 B/block
//   maxThreads  1024/block
//   104 CU @ 1.6 GHz
//
// 编译:
//   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 tutorial_i8_256.cu -o tut
// 运行:
//   ./tut check     正确性(对 CPU 参考)
//   ./tut bench     性能
//   ./tut prov      输出来源探针(把生产者身份 dump 成 csv)
// =============================================================================
#include <mc_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    mcError_t e = (x);                                                         \
    if (e != mcSuccess) {                                                      \
      fprintf(stderr, "[%s:%d] %s -> %s\n", __FILE__, __LINE__, #x,            \
              mcGetErrorString(e));                                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// 分块参数
//
//   BM x BN = 256 x 256   一个 block 负责的 C 子块
//   KTILE   = 256         题目要求的 K 方向宏 tile
//   BK      = 32          一级流水的 k 深度(LDS 里实际驻留的 k)
//                         KTILE/BK = 8 个 stage
//
//   线程数的推导(唯一解):
//     每 lane 的累加器数 = BM*BN/THREADS 个 int32。
//     THREADS=256 -> 256 个 int32/lane,远超 VGPR 预算,编不出来。
//     THREADS=512 -> 128 个,仍然爆。
//     THREADS=1024 -> 64 个 int32/lane,可行,且正好是 maxThreadsPerBlock。
//   所以 256x256 的 tile 在这张卡上**必须**用 1024 线程 = 16 个 wave。
//
//   16 个 wave 排成 4x4,每个 wave 负责 64x64 -> 4x4 个 16x16 的 MMA tile。
//   (排成 2x8 则 wave tile 为 128x32,累加器数不变但 A 片段复用率下降)
// -----------------------------------------------------------------------------
#define BM 256
#define BN 256
#define BK 32
#define KTILE 256

#define WAVE 64
#define WAVES_M 4
#define WAVES_N 4
#define THREADS (WAVES_M * WAVES_N * WAVE) // 1024

#define WM (BM / WAVES_M) // 64
#define WN (BN / WAVES_N) // 64
#define TM (WM / 16)      // 4  每 wave 在 M 向的 MMA tile 数
#define TN (WN / 16)      // 4  每 wave 在 N 向的 MMA tile 数

// -----------------------------------------------------------------------------
// 向量类型与 MMA
//
// SDK 原文(__clang_maca_mma_functions.h:2475):
//     d.x = __builtin_mxc_mma_16x16x16i8(a.x[0], b.x[0], c.x);
// a.x[0]/b.x[0] 的类型是 int —— fragment<...,signed char,...> 继承
// __frag_base<signed char,4,1>,storage 是 1 个 int:4 个 int8 塞进一个 32 位
// 寄存器。所以 A/B 片段在 C 语言层面就是一个 int。
// -----------------------------------------------------------------------------
typedef __attribute__((ext_vector_type(4))) int i32x4; // 累加器片段

__device__ __forceinline__ i32x4 mma_i8(int a, int b, i32x4 c) {
  return __builtin_mxc_mma_16x16x16i8(a, b, c);
}

// -----------------------------------------------------------------------------
// 共享内存布局与 swizzle
//
// As: [m][k]  256 x 32 int8 = 8 KB     (A 行主序,直接搬,不转置)
// Bs: [n][k]  256 x 32 int8 = 8 KB     (B 必须从 [k][n] 转成 [n][k])
// 合计 16 KB,单缓冲。
//
// 为什么 Bs 要存 [n][k]:
//   SDK 的 matrix_b/col_major 加载是
//       f.x[0] = *(int *)(p + col * ldm + row);     // 一条 4B 读
//   而 row_major 版本是 4 条独立的单字节读 + 拼 union。
//   col_major 的 B 等价于 B^T 行主序,即 [n][k]。所以存成 [n][k] 才能
//   让一个 lane 用**一条** 4B 读拿到 B[k0..k0+3][n]。
//
// swizzle:
//   一次 LDS 读里,lane&15 = 0..15 的 16 个 lane 读 16 个不同的 m、同一个 k0。
//   裸地址 = m*BK + k0,m 的步长 32B = 8 个 bank。
//   bank = (m*8 + k0/4) % 32  ->  m=0,4,8,12 全落在同一 bank,4 路冲突。
//   做 chunk 级 XOR:把行内第 c 个 4B chunk 换到第 c ^ (m & 7) 个位置,
//   bank = (m*8 + ((k0/4) ^ (m&7))) % 32,m=0..7 铺满 8 组、互不相同,
//   只剩 m 与 m+8 一对,降到 2 路。
//   (彻底消掉要把 slot 加宽到 8B 让 16 lane x 8B = 128B 正好铺满 32 bank,
//    那需要 k-step 交织,见教程第 7 节;本文件保留 2 路以换取可读性)
// -----------------------------------------------------------------------------
#define SWZ_CHUNKS (BK / 4) // 8 个 4B chunk
#define SWZ_MASK (SWZ_CHUNKS - 1)

#ifndef USE_SWIZZLE
#define USE_SWIZZLE 1
#endif

// 行 r(m 或 n)、列 k 的字节偏移
__device__ __host__ __forceinline__ int sidx(int r, int k) {
#if USE_SWIZZLE
  int c = k >> 2;             // 第几个 4B chunk
  int lo = k & 3;             // chunk 内偏移
  int cs = c ^ (r & SWZ_MASK); // 置换后的 chunk
  return r * BK + (cs << 2) + lo;
#else
  return r * BK + k;
#endif
}

// =============================================================================
// kernel
//   C(int32) = A(int8) * B(int8)         row-major A[M][K], B[K][N], C[M][N]
// =============================================================================
__global__ __launch_bounds__(THREADS) void gemm_i8_256(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  __shared__ int8_t As[BM * BK];
  __shared__ int8_t Bs[BN * BK];

  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int wm = wave / WAVES_N; // wave 在 M 向的坐标 0..3
  const int wn = wave % WAVES_N; // wave 在 N 向的坐标 0..3

  const int m0 = blockIdx.y * BM; // 本 block 的 C 子块原点
  const int n0 = blockIdx.x * BN;

  // --- 片段内的 lane -> 元素 映射(SDK 原文,见教程第 4 节) ---
  const int a_row = lane & 15;        // A 片段:m = lane&15
  const int b_col = lane & 15;        // B 片段:n = lane&15
  const int frag_k = (lane >> 4) * 4; // A/B 共用:k0 = (lane/16)*4
  const int c_m0 = (lane >> 4) * 4;   // 累加器:m = (lane/16)*4 + i
  const int c_n = lane & 15;          // 累加器:n = lane&15

  // --- 累加器:16 个片段 x 4 个 int32 = 64 个 int32 ---
  i32x4 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = (i32x4){0, 0, 0, 0};

  // --- 搬运分工 ---
  // A tile: 256 行 x 32B = 8192B,1024 线程 -> 8B/线程 -> 4 线程/行
  const int a_ld_row = tid >> 2;        // 0..255
  const int a_ld_k = (tid & 3) * 8;     // 0,8,16,24
  // B tile: 32 行(k) x 256B = 8192B -> 8B/线程 -> 32 线程/行
  const int b_ld_k = tid >> 5;          // 0..31
  const int b_ld_n = (tid & 31) * 8;    // 0..248

  for (int kt = 0; kt < K; kt += BK) {
    // ---- global -> LDS ----
    // A:行主序,8 个连续字节直接搬,k 方向连续,swizzle 只换 chunk 位置
    {
      const int8_t *src = A + (size_t)(m0 + a_ld_row) * K + kt + a_ld_k;
      // 8B = 2 个 chunk,各自 swizzle 后不一定相邻,所以分两次 4B 写
#pragma unroll
      for (int c = 0; c < 2; ++c)
        *(int *)&As[sidx(a_ld_row, a_ld_k + c * 4)] = *(const int *)(src + c * 4);
    }
    // B:读 8 个连续的 n(同一个 k),写到 8 个不同的 n 行 -> 转置,8 次单字节写
    {
      const int8_t *src = B + (size_t)(kt + b_ld_k) * N + n0 + b_ld_n;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        Bs[sidx(b_ld_n + e, b_ld_k)] = src[e];
    }
    __syncthreads();

    // ---- LDS -> 寄存器 -> MMA ----
    // BK=32 = 2 个 k-step(每条 MMA 吃 16 个 k)
#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      int af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *(const int *)&As[sidx(wm * WM + i * 16 + a_row, ks + frag_k)];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *(const int *)&Bs[sidx(wn * WN + j * 16 + b_col, ks + frag_k)];
      // 8 条 LDS 读换 16 条 MMA
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = mma_i8(af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // ---- store C ----
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int cm = m0 + wm * WM + i * 16 + c_m0;
      int cn = n0 + wn * WN + j * 16 + c_n;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        C[(size_t)(cm + e) * N + cn] = acc[i][j][e];
    }
}

// =============================================================================
// 来源探针 kernel
//   不算 GEMM,而是把每个 C 元素的"生产者身份"写出来:
//     wave id / lane id / 累加器 tile 号 (i,j) / 片段内元素号 e
//   这样可视化用的是真机实际执行出来的映射,而不是推演的图。
// =============================================================================
__global__ __launch_bounds__(THREADS) void prov_probe(int32_t *wave_id,
                                                      int32_t *lane_id,
                                                      int32_t *acc_ij,
                                                      int32_t *acc_e, int N) {
  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  const int c_m0 = (lane >> 4) * 4;
  const int c_n = lane & 15;

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int cm = wm * WM + i * 16 + c_m0;
      int cn = wn * WN + j * 16 + c_n;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        size_t o = (size_t)(cm + e) * N + cn;
        wave_id[o] = wave;
        lane_id[o] = lane;
        acc_ij[o] = i * TN + j;
        acc_e[o] = e;
      }
    }
}

// =============================================================================
// host
// =============================================================================
static void ref_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N,
                     int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k) s += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
      C[m * N + n] = s;
    }
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "check";
  int M = 256, N = 256, K = 256;
  if (argc > 4) { M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }

  size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
  int8_t *hA = (int8_t *)malloc(szA), *hB = (int8_t *)malloc(szB);
  int32_t *hC = (int32_t *)malloc(szC * 4), *hR = (int32_t *)malloc(szC * 4);

  srand(1234);
  for (size_t i = 0; i < szA; ++i) hA[i] = (int8_t)((rand() % 15) - 7);
  for (size_t i = 0; i < szB; ++i) hB[i] = (int8_t)((rand() % 15) - 7);

  int8_t *dA, *dB;
  int32_t *dC;
  CHECK(mcMalloc((void **)&dA, szA));
  CHECK(mcMalloc((void **)&dB, szB));
  CHECK(mcMalloc((void **)&dC, szC * 4));
  CHECK(mcMemcpy(dA, hA, szA, mcMemcpyHostToDevice));
  CHECK(mcMemcpy(dB, hB, szB, mcMemcpyHostToDevice));

  dim3 blk(THREADS), grd((N + BN - 1) / BN, (M + BM - 1) / BM);

  if (strcmp(mode, "prov") == 0) {
    // 只跑一个 block,dump 256x256 的生产者身份
    int32_t *d1, *d2, *d3, *d4;
    size_t sz = (size_t)BM * BN * 4;
    CHECK(mcMalloc((void **)&d1, sz));
    CHECK(mcMalloc((void **)&d2, sz));
    CHECK(mcMalloc((void **)&d3, sz));
    CHECK(mcMalloc((void **)&d4, sz));
    prov_probe<<<1, blk>>>(d1, d2, d3, d4, BN);
    CHECK(mcDeviceSynchronize());
    int32_t *h = (int32_t *)malloc(sz);
    const char *names[4] = {"wave.csv", "lane.csv", "accij.csv", "acce.csv"};
    int32_t *ds[4] = {d1, d2, d3, d4};
    for (int f = 0; f < 4; ++f) {
      CHECK(mcMemcpy(h, ds[f], sz, mcMemcpyDeviceToHost));
      FILE *fp = fopen(names[f], "w");
      for (int m = 0; m < BM; ++m) {
        for (int n = 0; n < BN; ++n)
          fprintf(fp, "%d%c", h[m * BN + n], n == BN - 1 ? '\n' : ',');
      }
      fclose(fp);
      printf("wrote %s\n", names[f]);
    }
    // 顺带打印几个 lane 的归属,便于正文引用
    printf("\n-- lane -> 片段元素 映射(SDK 公式) --\n");
    for (int l = 0; l < 64; l += 16)
      printf("lane %2d: A 片段 m=%2d k0=%2d | B 片段 n=%2d k0=%2d | C m0=%2d n=%2d\n",
             l, l & 15, (l >> 4) * 4, l & 15, (l >> 4) * 4, (l >> 4) * 4, l & 15);
    for (int l = 0; l < 4; ++l)
      printf("lane %2d: A 片段 m=%2d k0=%2d | B 片段 n=%2d k0=%2d | C m0=%2d n=%2d\n",
             l, l & 15, (l >> 4) * 4, l & 15, (l >> 4) * 4, (l >> 4) * 4, l & 15);
    return 0;
  }

  // 正确性
  CHECK(mcMemset(dC, 0, szC * 4));
  gemm_i8_256<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("launch error: %s\n", mcGetErrorString(le)); return 1; }
  CHECK(mcMemcpy(hC, dC, szC * 4, mcMemcpyDeviceToHost));

  if (strcmp(mode, "check") == 0) {
    ref_gemm(hA, hB, hR, M, N, K);
    size_t bad = 0;
    int32_t firstm = -1, firstn = -1;
    for (size_t i = 0; i < szC; ++i)
      if (hC[i] != hR[i]) {
        if (!bad) { firstm = (int)(i / N); firstn = (int)(i % N); }
        ++bad;
      }
    printf("M=%d N=%d K=%d  swizzle=%d\n", M, N, K, USE_SWIZZLE);
    if (bad)
      printf("FAIL  %zu/%zu mismatch, first at C[%d][%d] got %d want %d\n", bad,
             szC, firstm, firstn, hC[(size_t)firstm * N + firstn],
             hR[(size_t)firstm * N + firstn]);
    else
      printf("PASS  all %zu elements exact (int32 没有浮点误差问题)\n", szC);
    return bad ? 1 : 0;
  }

  // 性能
  {
    int iters = 200;
    for (int i = 0; i < 20; ++i) gemm_i8_256<<<grd, blk>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i) gemm_i8_256<<<grd, blk>>>(dA, dB, dC, M, N, K);
    CHECK(mcEventRecord(e1, 0));
    CHECK(mcEventSynchronize(e1));
    float ms = 0;
    CHECK(mcEventElapsedTime(&ms, e0, e1));
    double us = ms * 1000.0 / iters;
    double tops = 2.0 * M * N * K / (us * 1e-6) / 1e12;
    printf("M=%d N=%d K=%d swizzle=%d  grid=(%d,%d)  %.3f us  %.1f TOPS\n", M, N,
           K, USE_SWIZZLE, grd.x, grd.y, us, tops);
  }
  return 0;
}
