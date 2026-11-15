// =============================================================================
// mma_teach_256.cu  ——  C500 int8 MMA 教学版（M=N=256, K=256）
//
//   布局与 kernel_test_v1.cu 完全一致，只把工程手段换成最朴素的写法，
//   方便对照「谁负责哪块区域」：
//     · 异步 ldg_b128_bsm_predicator  ->  普通 16B 指针拷贝
//     · arrive_gvmcnt + barrier_inst  ->  __syncthreads()
//     · 下标算式、tile 尺寸、wave 排布、写回公式  ->  一个字没改
//
//   四层区域划分（权威定义就是最后那段写回）：
//     C 256x256
//       -> grid 2x2 = 4 个 block，每个 block 管 128x128
//          -> block 内 256 线程 = 4 个 wave，排成 2x2，每个 wave 管 64x64
//             -> wave 内 4x4 = 16 个 16x16 的 MMA tile
//                -> tile 内 64 条 lane，每条持有同一列的连续 4 行
//
//   编译：/opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_teach_256.cu -o mma_teach_256
//   运行：./mma_teach_256            正确性 + 层级来源表
//         ./mma_teach_256 164 81     拆解 C[164][81] 的完整来源链
//         ./mma_teach_256 bench      计时（可加 M N K）
// =============================================================================
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
  do {                                                                         \
    mcError_t e_ = (x);                                                        \
    if (e_ != mcSuccess) {                                                     \
      fprintf(stderr, "[%s:%d] %s -> %s\n", __FILE__, __LINE__, #x,            \
              mcGetErrorString(e_));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- 布局常量：逐项对应 kernel_test_v1.cu 的 gemm_config ----
#define TILE_M 128                   // kTileM
#define TILE_N 128                   // kTileN
#define TILE_K 256                   // kTileK
#define THREADS 256                  // kThreadNum
#define WAVE 64                      // kWarpSize
#define WAVE_M 2                     // kWaveM     block 内 wave 的 2x2 排布
#define WAVE_N 2                     // kWaveN
#define WM (TILE_M / WAVE_M)         // 64   一个 wave 负责的 M 范围
#define WN (TILE_N / WAVE_N)         // 64   一个 wave 负责的 N 范围
#define MMA_M (WM / 16)              // 4    wave 内 m 方向的 tile 数
#define MMA_N (WN / 16)              // 4    wave 内 n 方向的 tile 数
#define KCHUNK (TILE_K / 16 / 4)     // 4    kLdsNumPerK：一次 16B 读覆盖 4 个片段
#define LDG_ROW_THREADS 16           // kLdgThreadK：16 个线程合搬一行
#define SMEM_BYTES (2 * TILE_M * TILE_K)  // 64KB

typedef __attribute__((ext_vector_type(4))) int i32x4;  // 累加器：4 个 int32
// 注意：名字说的是「16 字节」，底层是 4 个 int。所以 af[m][c][w] 的 w=0..3
// 取的是第 w 个 4 字节片段，不是第 w 个字节。(kernel_test_v1.cu 那边直接写
// ABType=int32_t 的 a[..][..][4]，没有这个歧义)
typedef __attribute__((ext_vector_type(4))) int i8x16;  // 16B = 4 个 MMA 片段

// -----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS) void mma_teach_256(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  extern __shared__ int8_t smem[];
  int8_t *As = smem;                          // 128 x 256 = 32KB
  int8_t *Bs = smem + TILE_M * TILE_K;        // 128 x 256 = 32KB

  const int tid = threadIdx.x;
  const int wave = tid / WAVE;                // 0..3
  const int lane = tid % WAVE;                // 0..63
  const int wave_m = wave / WAVE_N;           // 0..1
  const int wave_n = wave % WAVE_N;           // 0..1

  const int block_row = blockIdx.y * TILE_M;  // 本 block 负责的 M 起点
  const int block_col = blockIdx.x * TILE_N;  // 本 block 负责的 N 起点

  i32x4 acc[MMA_M][MMA_N];
#pragma unroll
  for (int m = 0; m < MMA_M; ++m)
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) acc[m][n] = (i32x4){0, 0, 0, 0};

  // 搬运分工：16 个线程 × 16B = 256B = 一整行 TILE_K
  const int ld_row = tid / LDG_ROW_THREADS;         // 0..15  本轮搬第几行
  const int ld_k = (tid % LDG_ROW_THREADS) * 16;    // 0..240 该行第几个 16B

  // 取片段用的 lane 坐标
  const int mn = lane % 16;                   // tile 内：A 的行 / B 的列
  const int q = lane / 16;                    // 0..3    k 组

  for (int tk = 0; tk < K / TILE_K; ++tk) {   // K=256 时这个循环只走一轮
    // ---- 1. global -> LDS：每线程一次 16B，8 轮铺满 128 行 ----
#pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i) {
      const int r = ld_row + i * 16;          // 0..127
      const int koff = tk * TILE_K + ld_k;
      *(i8x16 *)&As[r * TILE_K + ld_k] =
          *(const i8x16 *)&A[(size_t)(block_row + r) * K + koff];
      *(i8x16 *)&Bs[r * TILE_K + ld_k] =
          *(const i8x16 *)&B[(size_t)(block_col + r) * K + koff];
    }
    __syncthreads();

    // ---- 2. LDS -> 寄存器：32 条 16B 读(A/B 各 16) -> 64+64 个片段 ----
    //   af[s][c]：s = wave 内第几个 16 行，c = 第几个 k 组
    //   每条读的 4 个 word 各是一个完整片段，分给 4 条不同的 MMA(w=0..3)
    i8x16 af[MMA_M][KCHUNK], bf[MMA_N][KCHUNK];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * 16 + c * 64;
        af[s][c] = *(const i8x16 *)&As[(wave_m * WM + s * 16 + mn) * TILE_K + k_off];
        bf[s][c] = *(const i8x16 *)&Bs[(wave_n * WN + s * 16 + mn) * TILE_K + k_off];
      }

    // ---- 3. MMA：4(m) × 4(n) × 4(k 组) × 4(word) = 256 条 ----
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
      for (int n = 0; n < MMA_N; ++n)
#pragma unroll
        for (int c = 0; c < KCHUNK; ++c)
#pragma unroll
          for (int w = 0; w < 4; ++w)
            acc[m][n] = __builtin_mxc_mma_16x16x16i8(af[m][c][w], bf[n][c][w],
                                                     acc[m][n]);
    __syncthreads();
  }

  // ---- 4. 写回：这 4 行就是「谁拥有哪个 C 元素」的权威定义 ----
  const int cm = (lane / 16) * 4;             // tile 内的行起点
  const int cn = lane % 16;                   // tile 内的列
#pragma unroll
  for (int m = 0; m < MMA_M; ++m)
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
      const int row = block_row + wave_m * WM + m * 16 + cm;
      const int col = block_col + wave_n * WN + n * 16 + cn;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        C[(size_t)(row + e) * N + col] = acc[m][n][e];
    }
}

// -----------------------------------------------------------------------------
// 来源探针：把「我是谁」编码进 C，用来把上面那个公式和硬件实际行为对账
//   code = (((bid * THREADS + tid) * MMA_M + m) * MMA_N + n) * 4 + e
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS) void prov_256(int32_t *__restrict__ C,
                                                    int N) {
  const int tid = threadIdx.x;
  const int wave = tid / WAVE, lane = tid % WAVE;
  const int wave_m = wave / WAVE_N, wave_n = wave % WAVE_N;
  const int block_row = blockIdx.y * TILE_M, block_col = blockIdx.x * TILE_N;
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  const int cm = (lane / 16) * 4, cn = lane % 16;
#pragma unroll
  for (int m = 0; m < MMA_M; ++m)
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) {
      const int row = block_row + wave_m * WM + m * 16 + cm;
      const int col = block_col + wave_n * WN + n * 16 + cn;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        C[(size_t)(row + e) * N + col] =
            (((bid * THREADS + tid) * MMA_M + m) * MMA_N + n) * 4 + e;
    }
}

// -----------------------------------------------------------------------------
// 读图探针：让每个线程把自己那 16 条 LDS 读的坐标写出来
//   out[tid*64 + (s*KCHUNK+c)*4 + {0,1,2,3}] = {As 行, 字节偏移, Bs 行, 字节偏移}
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS) void reads_256(int32_t *__restrict__ out) {
  const int tid = threadIdx.x;
  const int wave = tid / WAVE, lane = tid % WAVE;
  const int wave_m = wave / WAVE_N, wave_n = wave % WAVE_N;
  const int mn = lane % 16, q = lane / 16;
#pragma unroll
  for (int s = 0; s < MMA_M; ++s)
#pragma unroll
    for (int c = 0; c < KCHUNK; ++c) {
      const int i = tid * 64 + (s * KCHUNK + c) * 4;
      out[i + 0] = wave_m * WM + s * 16 + mn;  // As 行
      out[i + 1] = q * 16 + c * 64;            // 行内字节偏移
      out[i + 2] = wave_n * WN + s * 16 + mn;  // Bs 行
      out[i + 3] = q * 16 + c * 64;
    }
}

// -----------------------------------------------------------------------------
// 部分和探针：盯住一个 (bid, tid, m, n, e)，把每条 MMA 执行完之后累加器的读数
// 依次写出来。结构和主 kernel 逐行一样，只多了一个条件记录。
//   out[i] = 第 i 条 MMA 之后 acc[t_m][t_n][t_e] 的值
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS) void partial_256(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B, int K,
    int t_bid, int t_tid, int t_m, int t_n, int t_e,
    int32_t *__restrict__ out) {
  extern __shared__ int8_t smem[];
  int8_t *As = smem, *Bs = smem + TILE_M * TILE_K;

  const int tid = threadIdx.x;
  const int wave = tid / WAVE, lane = tid % WAVE;
  const int wave_m = wave / WAVE_N, wave_n = wave % WAVE_N;
  const int block_row = blockIdx.y * TILE_M, block_col = blockIdx.x * TILE_N;
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  const int watch = (bid == t_bid && tid == t_tid);

  i32x4 acc[MMA_M][MMA_N];
#pragma unroll
  for (int m = 0; m < MMA_M; ++m)
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) acc[m][n] = (i32x4){0, 0, 0, 0};

  const int ld_row = tid / LDG_ROW_THREADS;
  const int ld_k = (tid % LDG_ROW_THREADS) * 16;
  const int mn = lane % 16, q = lane / 16;
  int rec = 0;

  for (int tk = 0; tk < K / TILE_K; ++tk) {
#pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i) {
      const int r = ld_row + i * 16;
      const int koff = tk * TILE_K + ld_k;
      *(i8x16 *)&As[r * TILE_K + ld_k] =
          *(const i8x16 *)&A[(size_t)(block_row + r) * K + koff];
      *(i8x16 *)&Bs[r * TILE_K + ld_k] =
          *(const i8x16 *)&B[(size_t)(block_col + r) * K + koff];
    }
    __syncthreads();

    i8x16 af[MMA_M][KCHUNK], bf[MMA_N][KCHUNK];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * 16 + c * 64;
        af[s][c] = *(const i8x16 *)&As[(wave_m * WM + s * 16 + mn) * TILE_K + k_off];
        bf[s][c] = *(const i8x16 *)&Bs[(wave_n * WN + s * 16 + mn) * TILE_K + k_off];
      }

#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
      for (int n = 0; n < MMA_N; ++n)
#pragma unroll
        for (int c = 0; c < KCHUNK; ++c)
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            acc[m][n] = __builtin_mxc_mma_16x16x16i8(af[m][c][w], bf[n][c][w],
                                                     acc[m][n]);
            if (watch && m == t_m && n == t_n) out[rec++] = acc[m][n][t_e];
          }
    __syncthreads();
  }
}

// -----------------------------------------------------------------------------
// host 侧的闭式公式：给定 C 的全局坐标 (m, n)，反推是谁算的
// -----------------------------------------------------------------------------
typedef struct {
  int bid, by, bx;     // block
  int wave, wave_m, wave_n;
  int mm, nn;          // wave 内的 tile
  int lane, e;
  int tid;
} Owner;

static int g_grid_x = 1;  // = N / TILE_N，供 owner_of 复算 bid

static Owner owner_of(int m, int n) {
  Owner o;
  o.by = m / TILE_M;  o.bx = n / TILE_N;
  o.bid = o.by * g_grid_x + o.bx;
  const int im = m % TILE_M, in = n % TILE_N;
  o.wave_m = im / WM;  o.wave_n = in / WN;
  o.wave = o.wave_m * WAVE_N + o.wave_n;
  const int tm = im % WM, tn = in % WN;
  o.mm = tm / 16;  o.nn = tn / 16;
  const int lm = tm % 16, ln = tn % 16;
  o.lane = (lm / 4) * 16 + ln;
  o.e = lm % 4;
  o.tid = o.wave * WAVE + o.lane;
  return o;
}

static void ref_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N,
                     int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k)
        s += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)n * K + k];
      C[(size_t)m * N + n] = s;
    }
}

// ./mma_teach_256 reads <wave> <lane>  —— 打印该线程的 16 条 LDS 读
static int dump_reads(int wave, int lane) {
  const int tid = wave * WAVE + lane;
  int32_t *d, *hv = (int32_t *)malloc(THREADS * 64 * 4);
  CHECK(mcMalloc((void **)&d, (size_t)THREADS * 64 * 4));
  CHECK(mcMemset(d, 0, (size_t)THREADS * 64 * 4));
  reads_256<<<1, THREADS>>>(d);
  CHECK(mcDeviceSynchronize());
  CHECK(mcMemcpy(hv, d, (size_t)THREADS * 64 * 4, mcMemcpyDeviceToHost));
  printf("=== wave %d / lane %d (tid=%d) 的 16 条 LDS 读 ===\n", wave, lane, tid);
  printf("  mn = lane%%16 = %d    q = lane/16 = %d\n", lane % 16, lane / 16);
  printf("  s c | As 行  字节范围   逻辑 k      | Bs 行\n");
  for (int s = 0; s < MMA_M; ++s) {
    for (int c = 0; c < KCHUNK; ++c) {
      const int *v = hv + tid * 64 + (s * KCHUNK + c) * 4;
      printf("  %d %d |  %3d   %3d-%3d    %3d-%3d    |  %3d\n", s, c, v[0], v[1],
             v[1] + 15, v[1], v[1] + 15, v[2]);
    }
    if (s < MMA_M - 1) printf("      |\n");
  }
  printf("  -> As 只碰 4 个行号（间距 16）× 4 个字节块（间距 64），共 16 条\n");
  free(hv);
  return 0;
}

// 真机 dump：C[row][col] 这一个数，16 条 MMA 各加了多少
static int dump_partials(int row, int col) {
  const int M = 256, N = 256, K = 256;
  g_grid_x = N / TILE_N;
  const Owner o = owner_of(row, col);
  const size_t szA = (size_t)M * K, szB = (size_t)N * K;
  int8_t *hA = (int8_t *)malloc(szA), *hB = (int8_t *)malloc(szB);
  srand(1234);
  for (size_t i = 0; i < szA; ++i) hA[i] = (int8_t)((rand() % 15) - 7);
  for (size_t i = 0; i < szB; ++i) hB[i] = (int8_t)((rand() % 15) - 7);

  int8_t *dA, *dB;
  int32_t *dO;
  const int NREC = 16 * (K / TILE_K);
  CHECK(mcMalloc((void **)&dA, szA));
  CHECK(mcMalloc((void **)&dB, szB));
  CHECK(mcMalloc((void **)&dO, (size_t)NREC * 4));
  CHECK(mcMemcpy(dA, hA, szA, mcMemcpyHostToDevice));
  CHECK(mcMemcpy(dB, hB, szB, mcMemcpyHostToDevice));
  CHECK(mcMemset(dO, 0, (size_t)NREC * 4));
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  partial_256<<<grid, blk, SMEM_BYTES>>>(dA, dB, K, o.bid, o.tid, o.mm, o.nn,
                                         o.e, dO);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("launch error: %s\n", mcGetErrorString(le)); return 1; }
  int32_t *hv = (int32_t *)malloc((size_t)NREC * 4);
  CHECK(mcMemcpy(hv, dO, (size_t)NREC * 4, mcMemcpyDeviceToHost));

  long long ref = 0;
  for (int k = 0; k < K; ++k)
    ref += (long long)hA[(size_t)row * K + k] * (long long)hB[(size_t)col * K + k];

  printf("=== C[%d][%d] 的 16 条部分和（真机 dump）===\n", row, col);
  printf("  归属：block %d (bx=%d,by=%d) / wave %d / tile(%d,%d) / lane %d / e=%d\n",
         o.bid, o.bx, o.by, o.wave, o.mm, o.nn, o.lane, o.e);
  printf("   c  w |   这条加了多少 |   累加器变成\n");
  int prev = 0;
  for (int i = 0; i < NREC; ++i) {
    const int c = (i / 4) % 4, w = i % 4;
    printf("   %d  %d | %14d | %12d\n", c, w, hv[i] - prev, hv[i]);
    prev = hv[i];
  }
  printf("  最终 = %d   CPU 参考 = %lld   -> %s\n", hv[NREC - 1], ref,
         hv[NREC - 1] == ref ? "相等" : "不符");
  const int ok = (hv[NREC - 1] == ref);
  free(hA); free(hB); free(hv);
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  int M = 256, N = 256, K = 256;
  if (argc > 3 && strcmp(argv[1], "reads") == 0)
    return dump_reads(atoi(argv[2]), atoi(argv[3]));
  if (argc > 3 && strcmp(argv[1], "partial") == 0)
    return dump_partials(atoi(argv[2]), atoi(argv[3]));
  const int bench = (argc > 1 && strcmp(argv[1], "bench") == 0);
  if (bench && argc > 4) { M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }

  const size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
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

  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  g_grid_x = N / TILE_N;
  CHECK(mcMemset(dC, 0, szC * 4));
  mma_teach_256<<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) {
    printf("launch error: %s\n", mcGetErrorString(le));
    return 1;
  }
  CHECK(mcMemcpy(hC, dC, szC * 4, mcMemcpyDeviceToHost));

  // ---------------------------------------------------------------- bench ----
  if (bench) {
    ref_gemm(hA, hB, hR, M, N, K);
    size_t bad = 0;
    for (size_t i = 0; i < szC; ++i) if (hC[i] != hR[i]) ++bad;
    const int iters = 200;
    for (int i = 0; i < 20; ++i)
      mma_teach_256<<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_teach_256<<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcEventRecord(e1, 0));
    CHECK(mcEventSynchronize(e1));
    float ms = 0;
    CHECK(mcEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1000.0 / iters;
    printf("M=%d N=%d K=%d  %s  %d block  %.3f us  %.1f TOPS\n", M, N, K,
           bad ? "FAIL" : "PASS", grid.x * grid.y, us,
           2.0 * M * N * K / (us * 1e-6) / 1e12);
    return bad ? 1 : 0;
  }

  // ------------------------------------------------------------ 正确性 ----
  printf("=== 规模与分工 ===\n");
  printf("C = A(%dx%d) x B^T(%dx%d)\n", M, K, K, N);
  printf("block tile %dx%dx%d   grid %dx%d = %d 个 block\n", TILE_M, TILE_N,
         TILE_K, grid.y, grid.x, grid.x * grid.y);
  printf("block 内 %d 线程 = %d 个 wave，排成 %dx%d，每 wave 管 %dx%d\n",
         THREADS, THREADS / WAVE, WAVE_M, WAVE_N, WM, WN);
  printf("wave 内 %dx%d = %d 个 16x16 tile；每 wave %d 条 MMA\n", MMA_M, MMA_N,
         MMA_M * MMA_N, MMA_M * MMA_N * KCHUNK * 4);
  printf("LDS %d B = A %dx%d + B %dx%d\n", SMEM_BYTES, TILE_M, TILE_K, TILE_N,
         TILE_K);
  printf("全局 MMA 条数 %d  (= (%d/16)^3)\n",
         grid.x * grid.y * (THREADS / WAVE) * MMA_M * MMA_N * KCHUNK * 4, M);

  ref_gemm(hA, hB, hR, M, N, K);
  size_t bad = 0;
  for (size_t i = 0; i < szC; ++i) if (hC[i] != hR[i]) ++bad;
  printf("\n=== 正确性 ===\n%s  (%zu / %zu 不符)\n", bad ? "FAIL" : "PASS", bad,
         szC);

  // 探针会把 dC 覆盖掉，先把这一轮的 GPU 结果留一份
  int32_t *hG = (int32_t *)malloc(szC * 4);
  memcpy(hG, hC, szC * 4);

  // ------------------------------------------------- 公式 vs 硬件 对账 ----
  CHECK(mcMemset(dC, 0, szC * 4));
  prov_256<<<grid, blk, SMEM_BYTES>>>(dC, N);
  CHECK(mcDeviceSynchronize());
  CHECK(mcMemcpy(hC, dC, szC * 4, mcMemcpyDeviceToHost));
  size_t mism = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const Owner o = owner_of(m, n);
      const int want = (((o.bid * THREADS + o.tid) * MMA_M + o.mm) * MMA_N + o.nn) * 4 + o.e;
      if (hC[(size_t)m * N + n] != want) ++mism;
    }
  printf("\n=== 闭式公式 vs 硬件实际 ===\n%zu / %zu 不符%s\n", mism, szC,
         mism ? "" : "  -> 公式与硬件完全一致");

  // ------------------------------------------------------------- 表格 ----
  printf("\n=== 表1：哪个 block 负责哪块 (每格 %dx%d) ===\n", TILE_M, TILE_N);
  printf("        n:0-127  n:128-255\n");
  for (int by = 0; by < (int)grid.y; ++by) {
    printf("m:%3d-%3d", by * TILE_M, by * TILE_M + TILE_M - 1);
    for (int bx = 0; bx < (int)grid.x; ++bx)
      printf("   blk(%d,%d)", bx, by);
    printf("\n");
  }

  printf("\n=== 表2：block 内哪个 wave 负责哪块 (每格 %dx%d) ===\n", WM, WN);
  printf("          n:0-63   n:64-127\n");
  for (int wm = 0; wm < WAVE_M; ++wm) {
    printf("m:%3d-%3d", wm * WM, wm * WM + WM - 1);
    for (int wn = 0; wn < WAVE_N; ++wn)
      printf("   wave %d ", wm * WAVE_N + wn);
    printf("\n");
  }

  printf("\n=== 表3：wave 内 16 个 tile 的 (mm,nn) ===\n     ");
  for (int nn = 0; nn < MMA_N; ++nn) printf("  n:%2d-%2d", nn * 16, nn * 16 + 15);
  printf("\n");
  for (int mm = 0; mm < MMA_M; ++mm) {
    printf("m:%2d-%2d", mm * 16, mm * 16 + 15);
    for (int nn = 0; nn < MMA_N; ++nn) printf("   (%d,%d) ", mm, nn);
    printf("\n");
  }

  printf("\n=== 表4：一个 16x16 tile 内，哪条 lane 持有哪格 ===\n");
  printf("     ");
  for (int c = 0; c < 16; ++c) printf("%4d", c);
  printf("\n");
  for (int r = 0; r < 16; ++r) {
    printf("r=%2d ", r);
    for (int c = 0; c < 16; ++c) printf("%4d", (r / 4) * 16 + c);
    printf("\n");
  }
  printf("（格内是 lane 号；同一 lane 的 4 行用 acc 的 e=0..3 区分，e = 行%%4）\n");

  printf("\n=== 表5：16 组 (c,w) 各覆盖哪些 k ===\n");
  for (int c = 0; c < KCHUNK; ++c)
    for (int w = 0; w < 4; ++w) {
      printf("c=%d w=%d: k =", c, w);
      for (int q = 0; q < 4; ++q) {
        const int k0 = c * 64 + q * 16 + w * 4;
        printf("  %3d-%3d", k0, k0 + 3);
      }
      printf("   (q=0..3 四条 lane 组)\n");
    }
  printf("k = 64*c + 16*q + 4*w + j  —— 混合基数分解，0..255 各覆盖一次\n");

  // ------------------------------------------- 单个元素的完整来源链 ----
  if (argc > 2) {
    const int m = atoi(argv[1]), n = atoi(argv[2]);
    if (m >= 0 && m < M && n >= 0 && n < N) {
      const Owner o = owner_of(m, n);
      printf("\n=== 拆解 C[%d][%d] ===\n", m, n);
      printf("  block (x=%d, y=%d)   负责 m %d-%d, n %d-%d\n", o.bx, o.by,
             o.by * TILE_M, o.by * TILE_M + TILE_M - 1, o.bx * TILE_N,
             o.bx * TILE_N + TILE_N - 1);
      printf("  wave %d (wave_m=%d, wave_n=%d)   block 内偏移 m +%d, n +%d\n",
             o.wave, o.wave_m, o.wave_n, o.wave_m * WM, o.wave_n * WN);
      printf("  tile (mm=%d, nn=%d)   wave 内偏移 m +%d, n +%d\n", o.mm, o.nn,
             o.mm * 16, o.nn * 16);
      printf("  lane %d 的 acc[%d][%d][%d]   (block 内 tid=%d)\n", o.lane, o.mm,
             o.nn, o.e, o.tid);
      printf("  验算 行 = %d + %d*%d + %d*16 + (%d/16)*4 + %d = %d\n",
             o.by * TILE_M, o.wave_m, WM, o.mm, o.lane, o.e,
             o.by * TILE_M + o.wave_m * WM + o.mm * 16 + (o.lane / 16) * 4 + o.e);
      printf("  验算 列 = %d + %d*%d + %d*16 + %d%%16 = %d\n", o.bx * TILE_N,
             o.wave_n, WN, o.nn, o.lane,
             o.bx * TILE_N + o.wave_n * WN + o.nn * 16 + o.lane % 16);
      long long ref = 0;
      for (int k = 0; k < K; ++k)
        ref += (long long)hA[(size_t)m * K + k] * (long long)hB[(size_t)n * K + k];
      printf("  CPU 参考 = %lld，GPU = %d  -> %s\n", ref, hG[(size_t)m * N + n],
             ref == hG[(size_t)m * N + n] ? "相等" : "不等");
    }
  }
  return bad || mism ? 1 : 0;
}
