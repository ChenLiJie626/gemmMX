// =============================================================================
// tutorial_i8_256_v2.cu
//   沐曦 C500 (xcore1000) int8 -> int32 MMA,TileMNK = 256x256x256
//   v2:修掉 v1 的两个真实瓶颈
//
//   v1 的问题(实测 4096^3 只有 27.9 TOPS):
//     (1) B 转置用了 8 次单字节 LDS 写 / 线程 / stage。字节写无法合并,
//         且 8 个目标地址相距 BK=32 字节,逐个打到同一组 bank。
//     (2) BK=32 时 XOR swizzle 只能降到 2 路冲突(m 与 m+8 同 bank)。
//
//   v2 的两处改动:
//     (1) B 走寄存器内 4x4 转置:每线程读 4 个 k 行、各 4 个连续 n(4B 全局读,
//         64 个线程合起来 256B 连续),在寄存器里转置成 4 个 n、各 4 个连续 k,
//         写出 4 次 4B LDS 写。单字节写 8 次 -> dword 写 4 次。
//     (2) BK 32 -> 64。这一步让 swizzle 变成**完全无冲突**:
//         行步长 BK=64B = 16 个 bank,bank = (r*16 + ((k/4)^(r&15))) % 32。
//         r 为偶数时 (r*16)%32=0,落在 bank[0,16);r 为奇数落在 bank[16,32)。
//         16 个 lane 的 r&15 恰好取遍 0..15 -> 偶 r 的 8 个 c^(r&15) 互不相同,
//         奇 r 的 8 个同理 -> 16 个 lane 占 16 个不同 bank,零冲突。
//         代价:LDS 从 16KB 涨到 32KB(两个 tile 各 256x64),每 CU 仍可驻留
//         2 个 block = 2048 线程 = maxThreadsPerMultiProcessor,不损并发。
//         收益:每 stage 的 MMA 数从 32 涨到 64,搬运指令摊薄一半。
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

#define BM 256
#define BN 256
#ifndef BK
#define BK 64
#endif
#define KTILE 256

#define WAVE 64
#define WAVES_M 4
#define WAVES_N 4
#define THREADS (WAVES_M * WAVES_N * WAVE) // 1024
#define WM (BM / WAVES_M)                  // 64
#define WN (BN / WAVES_N)                  // 64
#define TM (WM / 16)                       // 4
#define TN (WN / 16)                       // 4

typedef __attribute__((ext_vector_type(4))) int i32x4;

__device__ __forceinline__ i32x4 mma_i8(int a, int b, i32x4 c) {
  return __builtin_mxc_mma_16x16x16i8(a, b, c);
}

#define SWZ_CHUNKS (BK / 4)
#define SWZ_MASK (SWZ_CHUNKS - 1)

#ifndef USE_SWIZZLE
#define USE_SWIZZLE 1
#endif

__device__ __host__ __forceinline__ int sidx(int r, int k) {
#if USE_SWIZZLE
  int c = k >> 2, lo = k & 3;
  return r * BK + (((c ^ (r & SWZ_MASK))) << 2) + lo;
#else
  return r * BK + k;
#endif
}

// =============================================================================
__global__ __launch_bounds__(THREADS) void gemm_i8_256_v2(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  __shared__ int8_t As[BM * BK];
  __shared__ int8_t Bs[BN * BK];

  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  const int a_row = lane & 15;
  const int b_col = lane & 15;
  const int frag_k = (lane >> 4) * 4;
  const int c_m0 = (lane >> 4) * 4;
  const int c_n = lane & 15;

  i32x4 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (i32x4){0, 0, 0, 0};

  // --- A 搬运分工:tile 256 行 x BK 字节,每线程 BK*256/1024 = 16B ---
  const int A_PT = BM * BK / THREADS;   // 16
  const int a_ld_row = tid / (BK / A_PT);
  const int a_ld_k = (tid % (BK / A_PT)) * A_PT;

  // --- B 搬运分工:把 tile 切成 4k x 4n 的微块,每线程领一个或多个 ---
  //   微块总数 = (BK/4) * (BN/4)。BK=64 时 = 16*64 = 1024 = THREADS,每线程正好
  //   一个;BK=32 时只有 512 个,所以这里必须按 THREADS 跨步循环,不能假设 1:1。
  //   (写死 1:1 会让后一半线程算出越界的 k,直接踩出 ATU fault —— 真踩过)
  const int b_t4 = BN / 4;              // 64
  const int B_TILES = (BK / 4) * (BN / 4);

  for (int kt = 0; kt < K; kt += BK) {
    // ---- A: global 连续 16B -> 4 次 4B LDS 写(swizzle 打散 chunk) ----
    {
      const int8_t *src = A + (size_t)(m0 + a_ld_row) * K + kt + a_ld_k;
#pragma unroll
      for (int c = 0; c < A_PT / 4; ++c)
        *(int *)&As[sidx(a_ld_row, a_ld_k + c * 4)] = *(const int *)(src + c * 4);
    }

    // ---- B: 4 次 4B 全局读 -> 寄存器内转置 -> 4 次 4B LDS 写 ----
    //  读:v[e] = B[k4+e][n4..n4+3]      (沿 n 连续,线程间也连续 -> 合并)
    //  写:w[j] = B[k4..k4+3][n4+j]      (沿 k 连续 -> 一条 dword 写)
#pragma unroll
    for (int t = tid; t < B_TILES; t += THREADS) {
      const int b_k4 = (t / b_t4) * 4;
      const int b_n4 = (t % b_t4) * 4;
      int v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[e] = *(const int *)(B + (size_t)(kt + b_k4 + e) * N + n0 + b_n4);
      // 字节级转置:w[j] 的第 e 个字节 = v[e] 的第 j 个字节
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int w = ((v[0] >> (j * 8)) & 0xff) | (((v[1] >> (j * 8)) & 0xff) << 8) |
                (((v[2] >> (j * 8)) & 0xff) << 16) |
                (((v[3] >> (j * 8)) & 0xff) << 24);
        *(int *)&Bs[sidx(b_n4 + j, b_k4)] = w;
      }
    }
    __syncthreads();

    // ---- MMA:BK/16 个 k-step,每步 8 条 LDS 读换 16 条 MMA ----
#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      int af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *(const int *)&As[sidx(wm * WM + i * 16 + a_row, ks + frag_k)];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *(const int *)&Bs[sidx(wn * WN + j * 16 + b_col, ks + frag_k)];
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
static void ref_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N,
                     int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t s = 0;
      for (int k = 0; k < K; ++k)
        s += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
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
  CHECK(mcMemset(dC, 0, szC * 4));
  gemm_i8_256_v2<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("launch error: %s\n", mcGetErrorString(le)); return 1; }
  CHECK(mcMemcpy(hC, dC, szC * 4, mcMemcpyDeviceToHost));

  if (strcmp(mode, "check") == 0) {
    ref_gemm(hA, hB, hR, M, N, K);
    size_t bad = 0;
    for (size_t i = 0; i < szC; ++i) if (hC[i] != hR[i]) ++bad;
    printf("v2 M=%d N=%d K=%d BK=%d swizzle=%d  %s (%zu bad)\n", M, N, K, BK,
           USE_SWIZZLE, bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
  }

  int iters = 200;
  for (int i = 0; i < 20; ++i) gemm_i8_256_v2<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcEvent_t e0, e1;
  CHECK(mcEventCreate(&e0));
  CHECK(mcEventCreate(&e1));
  CHECK(mcEventRecord(e0, 0));
  for (int i = 0; i < iters; ++i) gemm_i8_256_v2<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcEventRecord(e1, 0));
  CHECK(mcEventSynchronize(e1));
  float ms = 0;
  CHECK(mcEventElapsedTime(&ms, e0, e1));
  double us = ms * 1000.0 / iters;
  printf("v2 M=%d N=%d K=%d BK=%d swizzle=%d  %.3f us  %.1f TOPS\n", M, N, K, BK,
         USE_SWIZZLE, us, 2.0 * M * N * K / (us * 1e-6) / 1e12);
  return 0;
}
