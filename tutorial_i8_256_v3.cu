// =============================================================================
// tutorial_i8_256_v3.cu
//   沐曦 C500 (xcore1000) int8 -> int32 MMA,TileMNK = 256x256x256
//   v3 = v2 + 全局 -> 寄存器预取(把访存延迟藏到计算后面)
//
//   v2 的结构问题:单缓冲 LDS,一轮迭代是
//       global 读 -> LDS 写 -> sync -> MMA -> sync
//   global 读的延迟(数百~上千周期)完全暴露在关键路径上。而 256x256 的 tile
//   需要 1024 线程 + 32KB LDS,每个 CU 只能驻留 2 个 block,靠并发掩盖不了。
//
//   v3 把顺序改成:
//       LDS 写(上一轮预取的寄存器) -> sync -> 发下一轮的 global 读 -> MMA -> sync
//   下一轮的 global 读在 MMA 之前**发射**,然后 64 条 MMA 的时间用来等它回来。
//   代价只有 8 个寄存器(4 个 A word + 4 个转置后的 B word)。
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
#if BK != 64
#error "v3 的预取分工按 BK=64 推导(B 微块数正好 = THREADS),改 BK 请看 v2"
#endif

#define WAVE 64
#define WAVES_M 4
#define WAVES_N 4
#define THREADS (WAVES_M * WAVES_N * WAVE) // 1024
#define WM (BM / WAVES_M)
#define WN (BN / WAVES_N)
#define TM (WM / 16)
#define TN (WN / 16)

typedef __attribute__((ext_vector_type(4))) int i32x4;

__device__ __forceinline__ i32x4 mma_i8(int a, int b, i32x4 c) {
  return __builtin_mxc_mma_16x16x16i8(a, b, c);
}

#define SWZ_MASK (BK / 4 - 1)
#ifndef USE_SWIZZLE
#define USE_SWIZZLE 1
#endif

__device__ __forceinline__ int sidx(int r, int k) {
#if USE_SWIZZLE
  int c = k >> 2, lo = k & 3;
  return r * BK + (((c ^ (r & SWZ_MASK))) << 2) + lo;
#else
  return r * BK + k;
#endif
}

__global__ __launch_bounds__(THREADS) void gemm_i8_256_v3(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  __shared__ int8_t As[BM * BK];
  __shared__ int8_t Bs[BN * BK];

  const int tid = threadIdx.x;
  const int lane = tid & (WAVE - 1);
  const int wave = tid / WAVE;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  const int a_row = lane & 15, b_col = lane & 15;
  const int frag_k = (lane >> 4) * 4;
  const int c_m0 = (lane >> 4) * 4, c_n = lane & 15;

  i32x4 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (i32x4){0, 0, 0, 0};

  const int a_ld_row = tid / 4;          // 0..255
  const int a_ld_k = (tid % 4) * 16;     // 0,16,32,48
  const int b_k4 = (tid / (BN / 4)) * 4; // 0..60
  const int b_n4 = (tid % (BN / 4)) * 4; // 0..252

  const int8_t *aptr = A + (size_t)(m0 + a_ld_row) * K + a_ld_k;
  const int8_t *bptr = B + (size_t)b_k4 * N + n0 + b_n4;

  int ra[4], rb[4];

  // ---- 预取宏:global -> 寄存器(B 顺手在寄存器里转置好) ----
#define PREFETCH(kt)                                                           \
  do {                                                                         \
    const int8_t *as = aptr + (kt);                                            \
    _Pragma("unroll") for (int c = 0; c < 4; ++c)                              \
        ra[c] = *(const int *)(as + c * 4);                                    \
    int v[4];                                                                  \
    const int8_t *bs = bptr + (size_t)(kt) * N;                                \
    _Pragma("unroll") for (int e = 0; e < 4; ++e)                              \
        v[e] = *(const int *)(bs + (size_t)e * N);                             \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) rb[j] =                      \
        ((v[0] >> (j * 8)) & 0xff) | (((v[1] >> (j * 8)) & 0xff) << 8) |       \
        (((v[2] >> (j * 8)) & 0xff) << 16) |                                   \
        (((v[3] >> (j * 8)) & 0xff) << 24);                                    \
  } while (0)

  PREFETCH(0);

  for (int kt = 0; kt < K; kt += BK) {
    // ---- 寄存器 -> LDS ----
#pragma unroll
    for (int c = 0; c < 4; ++c)
      *(int *)&As[sidx(a_ld_row, a_ld_k + c * 4)] = ra[c];
#pragma unroll
    for (int j = 0; j < 4; ++j) *(int *)&Bs[sidx(b_n4 + j, b_k4)] = rb[j];
    __syncthreads();

    // ---- 先发下一轮的 global 读,再做 MMA,让 64 条 MMA 盖住访存延迟 ----
    if (kt + BK < K) PREFETCH(kt + BK);

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
#undef PREFETCH

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
  gemm_i8_256_v3<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("launch error: %s\n", mcGetErrorString(le)); return 1; }
  CHECK(mcMemcpy(hC, dC, szC * 4, mcMemcpyDeviceToHost));

  if (strcmp(mode, "check") == 0) {
    ref_gemm(hA, hB, hR, M, N, K);
    size_t bad = 0;
    for (size_t i = 0; i < szC; ++i) if (hC[i] != hR[i]) ++bad;
    printf("v3 M=%d N=%d K=%d BK=%d  %s (%zu bad)\n", M, N, K, BK,
           bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
  }

  int iters = 200;
  for (int i = 0; i < 20; ++i) gemm_i8_256_v3<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcEvent_t e0, e1;
  CHECK(mcEventCreate(&e0));
  CHECK(mcEventCreate(&e1));
  CHECK(mcEventRecord(e0, 0));
  for (int i = 0; i < iters; ++i) gemm_i8_256_v3<<<grd, blk>>>(dA, dB, dC, M, N, K);
  CHECK(mcEventRecord(e1, 0));
  CHECK(mcEventSynchronize(e1));
  float ms = 0;
  CHECK(mcEventElapsedTime(&ms, e0, e1));
  double us = ms * 1000.0 / iters;
  printf("v3 M=%d N=%d K=%d BK=%d  %.3f us  %.1f TOPS\n", M, N, K, BK, us,
         2.0 * M * N * K / (us * 1e-6) / 1e12);
  return 0;
}
