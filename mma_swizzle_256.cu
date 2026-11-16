// =============================================================================
// mma_swizzle_256.cu  ——  给 mma_teach_256.cu 加 XOR swizzle，消 LDS bank 冲突
//
//   除了 LDS 的寻址函数，这个文件和 mma_teach_256.cu 一个字不差：
//   tile 尺寸、wave 排布、搬运分工、MMA 顺序、写回公式全部照抄。
//   唯一的变量是模板参数 SW —— false 走原始布局，true 走 XOR 布局，
//   两个 kernel 由同一份源码实例化，所以 bench 是严格的同条件对照。
//
//   ---- 冲突在哪 ----
//   LDS 有 32 个 bank，每个 bank 4 字节，一拍最多吐 32 * 4 = 128 字节。
//   一行 TILE_K = 256 字节，正好是 128 的整数倍 —— 每一行都从 bank 0 开始。
//
//   取片段那一步（第 2 段）是这么读的：
//       行       = wave_m*64 + s*16 + mn    mn = lane%16  -> 16 条 lane 读 16 个不同的行
//       行内偏移 = q*16 + c*64              q  = lane/16  -> 同一个 q 的 16 条 lane 偏移相同
//   行不同、行内偏移相同 => 地址相差 256 字节的整数倍 => 落在同一组 bank。
//   一条读指令里 64 条 lane 只用到 16 个 bank，每个 bank 要排 16 个 dword，
//   要 16 拍才吐得完；而 64 * 16B / 128B = 8 拍才是下限。整整慢一倍。
//
//   ---- XOR 怎么修 ----
//   一行 256B = 16 个 16B 块，而一"轮" bank 只装得下 128B = 8 个块。
//   把第 j 个块挪到第 (j ^ (r & 7)) 个槽位：相邻 8 行的同一个 j 被打散到
//   8 个不同的槽位，于是 16 条 lane 的 bank 组取遍 0..7（各两次），
//   256 个 dword 均摊到 32 个 bank = 每 bank 8 个 = 正好 8 拍，打到下限。
//
//   XOR 的三个好处：
//     1. 是双射（异或同一个数两次还原），每行 16 个块原地重排，不占额外空间；
//     2. 只动块编号的低 3 位，访问仍然 16B 对齐、仍然是一条整块读，不拆指令；
//     3. 读和写用同一个函数，写路径的 bank 分布没被破坏（本来就在下限上）。
//
//   编译：/opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_swizzle_256.cu -o mma_swizzle_256
//   运行：./mma_swizzle_256            两个布局各验一次正确性
//         ./mma_swizzle_256 banks      打印 bank 直方图（纯 host 计算，不用 GPU）
//         ./mma_swizzle_256 bench      同条件对照计时（可加 M N K）
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

// ---- 布局常量：与 mma_teach_256.cu / kernel_test_v1.cu 完全一致 ----
#define TILE_M 128
#define TILE_N 128
#define TILE_K 256
#define THREADS 256
#define WAVE 64
#define WAVE_M 2
#define WAVE_N 2
#define WM (TILE_M / WAVE_M)              // 64
#define WN (TILE_N / WAVE_N)              // 64
#define MMA_M (WM / 16)                   // 4
#define MMA_N (WN / 16)                   // 4
#define KCHUNK (TILE_K / 16 / 4)          // 4
#define LDG_ROW_THREADS 16
#define SMEM_BYTES (2 * TILE_M * TILE_K)  // 64KB

// ---- bank 参数 ----
#define BANKS 32                          // LDS bank 数
#define BANK_B 4                          // 每个 bank 的宽度
#define CHUNK 16                          // 一次 LDS 访问的宽度
#define TURN_B (BANKS * BANK_B)           // 128  一拍吐得出的字节数
#define CHUNKS_PER_TURN (TURN_B / CHUNK)  // 8    一拍装得下几个 16B 块
#define SWZ_MASK (CHUNKS_PER_TURN - 1)    // 7

// ---- 寻址：逻辑 (行 r, 行内字节 k) -> LDS 字节下标。k 必须 16B 对齐 ----
//   用宏而不是 inline 函数，是为了让 SW 这个编译期常量把分支整个折掉。
#define NOSWZ(r, k) ((r) * TILE_K + (k))
#define SWZ(r, k)                                                              \
  ((r) * TILE_K + (((((k) >> 4) ^ ((r) & SWZ_MASK)))<< 4) + ((k) & (CHUNK - 1)))
#define LDS_AT(SW, r, k) ((SW) ? SWZ(r, k) : NOSWZ(r, k))

typedef __attribute__((ext_vector_type(4))) int i32x4;  // 累加器：4 个 int32
typedef __attribute__((ext_vector_type(4))) int i8x16;  // 16B = 4 个 MMA 片段

// -----------------------------------------------------------------------------
//  SW = false -> 原始布局（= mma_teach_256.cu）
//  SW = true  -> XOR 布局
// -----------------------------------------------------------------------------
template <bool SW>
__global__ __launch_bounds__(THREADS) void mma_256(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  extern __shared__ int8_t smem[];
  int8_t *As = smem;
  int8_t *Bs = smem + TILE_M * TILE_K;

  const int tid = threadIdx.x;
  const int wave = tid / WAVE;
  const int lane = tid % WAVE;
  const int wave_m = wave / WAVE_N;
  const int wave_n = wave % WAVE_N;

  const int block_row = blockIdx.y * TILE_M;
  const int block_col = blockIdx.x * TILE_N;

  i32x4 acc[MMA_M][MMA_N];
#pragma unroll
  for (int m = 0; m < MMA_M; ++m)
#pragma unroll
    for (int n = 0; n < MMA_N; ++n) acc[m][n] = (i32x4){0, 0, 0, 0};

  const int ld_row = tid / LDG_ROW_THREADS;
  const int ld_k = (tid % LDG_ROW_THREADS) * 16;

  const int mn = lane % 16;
  const int q = lane / 16;

  for (int tk = 0; tk < K / TILE_K; ++tk) {
    // ---- 1. global -> LDS：全局侧一个字没改，只有落点经过 swizzle ----
#pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i) {
      const int r = ld_row + i * 16;
      const int koff = tk * TILE_K + ld_k;
      *(i8x16 *)&As[LDS_AT(SW, r, ld_k)] =
          *(const i8x16 *)&A[(size_t)(block_row + r) * K + koff];
      *(i8x16 *)&Bs[LDS_AT(SW, r, ld_k)] =
          *(const i8x16 *)&B[(size_t)(block_col + r) * K + koff];
    }
    __syncthreads();

    // ---- 2. LDS -> 寄存器：冲突就出在这 32 条读上 ----
    i8x16 af[MMA_M][KCHUNK], bf[MMA_N][KCHUNK];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * 16 + c * 64;
        const int ra = wave_m * WM + s * 16 + mn;
        const int rb = wave_n * WN + s * 16 + mn;
        af[s][c] = *(const i8x16 *)&As[LDS_AT(SW, ra, k_off)];
        bf[s][c] = *(const i8x16 *)&Bs[LDS_AT(SW, rb, k_off)];
      }

    // ---- 3. MMA：256 条。swizzle 只换了数据从哪个地址取，配对关系没动 ----
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

  // ---- 4. 写回：与 mma_teach_256.cu 逐字相同 ----
  const int cm = (lane / 16) * 4;
  const int cn = lane % 16;
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

// =============================================================================
//  host 侧 bank 模型：把上面那两条寻址公式原样搬到 CPU 上数 bank。
//  不需要 GPU，也不需要 profiler —— bank 分布是地址的确定性函数。
// =============================================================================
static int lds_at(int sw, int r, int k) {
  return sw ? SWZ(r, k) : NOSWZ(r, k);
}

// 一次 16B 访问压 4 个连续 dword，落在 4 个连续 bank 上
static void add_access(int *hist, int byte_addr) {
  for (int d = 0; d < CHUNK / BANK_B; ++d)
    hist[(byte_addr / BANK_B + d) % BANKS]++;
}

// 第 2 段的一条读指令：64 条 lane，(base, s, c) 固定
static void read_hist(int sw, int base, int s, int c, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int lane = 0; lane < WAVE; ++lane) {
    const int r = base + s * 16 + (lane % 16);
    const int k = (lane / 16) * 16 + c * 64;
    add_access(hist, lds_at(sw, r, k));
  }
}

// 第 1 段的一条写指令：64 条 lane，i 固定
static void write_hist(int sw, int i, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int t = 0; t < WAVE; ++t)
    add_access(hist, lds_at(sw, t / LDG_ROW_THREADS + i * 16,
                            (t % LDG_ROW_THREADS) * 16));
}

static int hist_max(const int *h) {
  int m = 0;
  for (int b = 0; b < BANKS; ++b)
    if (h[b] > m) m = h[b];
  return m;
}

static int hist_used(const int *h) {
  int u = 0;
  for (int b = 0; b < BANKS; ++b)
    if (h[b]) ++u;
  return u;
}

// tag 的显示宽度约定为 4 列（"原始" / "XOR "），这样三行表头能对齐
static void print_hist(const char *tag, const int *h) {
  printf("  %s", tag);
  for (int b = 0; b < BANKS; ++b) printf("%3d", h[b]);
  printf("   用到 %2d/%d bank，最深 %2d -> %2d 拍\n", hist_used(h), BANKS,
         hist_max(h), hist_max(h));
}

static void bank_report(void) {
  const int floor_cycles = WAVE * CHUNK / TURN_B;  // 64*16/128 = 8
  int h[BANKS];

  printf("LDS：%d 个 bank × %d B = %d B/拍；一行 TILE_K = %d B = %d 轮\n", BANKS,
         BANK_B, TURN_B, TILE_K, TILE_K / TURN_B);
  printf("一条 wave 宽的 16B 访问要吐 %d × %d = %d B，下限 %d 拍\n\n", WAVE,
         CHUNK, WAVE * CHUNK, floor_cycles);

  printf("=== 第 2 段，一条 LDS 读指令 (base=0, s=0, c=0) 的 bank 深度 ===\n");
  printf("  bank");
  for (int b = 0; b < BANKS; ++b) printf("%3d", b);
  printf("\n");
  read_hist(0, 0, 0, 0, h);
  print_hist("原始", h);
  read_hist(1, 0, 0, 0, h);
  print_hist("XOR ", h);

  printf("\n=== 遍历全部 (base, s, c) / i 组合，取最坏 ===\n");
  for (int sw = 0; sw < 2; ++sw) {
    int worst = 0;
    for (int base = 0; base <= WM; base += WM)
      for (int s = 0; s < MMA_M; ++s)
        for (int c = 0; c < KCHUNK; ++c) {
          read_hist(sw, base, s, c, h);
          if (hist_max(h) > worst) worst = hist_max(h);
        }
    printf("  %-4s 读：最坏 %2d 拍（下限 %d）-> %s\n", sw ? "XOR" : "原始",
           worst, floor_cycles, worst == floor_cycles ? "打到下限" : "有冲突");
  }
  for (int sw = 0; sw < 2; ++sw) {
    int worst = 0;
    for (int i = 0; i < TILE_M / 16; ++i) {
      write_hist(sw, i, h);
      if (hist_max(h) > worst) worst = hist_max(h);
    }
    printf("  %-4s 写：最坏 %2d 拍（下限 %d）-> %s\n", sw ? "XOR" : "原始",
           worst, floor_cycles, worst == floor_cycles ? "打到下限" : "有冲突");
  }

  printf("\n=== 每个 k 窗口、每个 wave 的 LDS 读开销 ===\n");
  const int ninst = 2 * MMA_M * KCHUNK;  // A 16 条 + B 16 条
  printf("  原始 %d 条读 × 16 拍 = %4d 拍\n", ninst, ninst * 16);
  printf("  XOR  %d 条读 ×  8 拍 = %4d 拍   省掉 %d 拍\n", ninst, ninst * 8,
         ninst * 8);

  printf("\n=== swizzle 是双射吗（每行 16 个块原地重排，不越界不重叠）===\n");
  int bad = 0;
  for (int r = 0; r < TILE_M; ++r) {
    int seen[TILE_K / CHUNK];
    memset(seen, 0, sizeof(seen));
    for (int j = 0; j < TILE_K / CHUNK; ++j) {
      const int a = SWZ(r, j * CHUNK);
      if (a < r * TILE_K || a >= (r + 1) * TILE_K) {
        ++bad;
        continue;
      }
      if (seen[(a - r * TILE_K) / CHUNK]++) ++bad;
    }
  }
  printf("  %d 行 × %d 块：%s\n", TILE_M, TILE_K / CHUNK,
         bad ? "失败" : "无越界、无重叠，通过");
}

// =============================================================================
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

struct Run {
  double us;
  size_t bad;
};

template <bool SW>
static Run run_one(const int8_t *dA, const int8_t *dB, int32_t *dC,
                   const int32_t *hR, int32_t *hC, int M, int N, int K,
                   int iters) {
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  Run r;
  r.us = 0;
  r.bad = 0;

  CHECK(mcMemset(dC, 0, (size_t)M * N * 4));
  mma_256<SW><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) {
    fprintf(stderr, "launch error: %s\n", mcGetErrorString(le));
    exit(1);
  }
  CHECK(mcMemcpy(hC, dC, (size_t)M * N * 4, mcMemcpyDeviceToHost));

  if (hR)
    for (size_t i = 0; i < (size_t)M * N; ++i)
      if (hC[i] != hR[i]) ++r.bad;

  if (iters) {
    for (int i = 0; i < 20; ++i)
      mma_256<SW><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_256<SW><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcEventRecord(e1, 0));
    CHECK(mcEventSynchronize(e1));
    float ms = 0;
    CHECK(mcEventElapsedTime(&ms, e0, e1));
    r.us = ms * 1000.0 / iters;
    CHECK(mcEventDestroy(e0));
    CHECK(mcEventDestroy(e1));
  }
  return r;
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "banks") == 0) {
    bank_report();
    return 0;
  }

  const int bench = (argc > 1 && strcmp(argv[1], "bench") == 0);
  int M = 256, N = 256, K = 256;
  if (bench && argc > 4) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
  }
  if (M % TILE_M || N % TILE_N || K % TILE_K) {
    fprintf(stderr, "M/N 需为 %d 的倍数，K 需为 %d 的倍数\n", TILE_M, TILE_K);
    return 1;
  }

  const size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
  int8_t *hA = (int8_t *)malloc(szA), *hB = (int8_t *)malloc(szB);
  int32_t *hC = (int32_t *)malloc(szC * 4), *hR = NULL;
  srand(1234);
  for (size_t i = 0; i < szA; ++i) hA[i] = (int8_t)((rand() % 15) - 7);
  for (size_t i = 0; i < szB; ++i) hB[i] = (int8_t)((rand() % 15) - 7);

  // 大尺寸下 CPU 参考太慢，只在 <= 1024 时算
  const int verify = (M <= 1024 && N <= 1024 && K <= 1024);
  if (verify) {
    hR = (int32_t *)malloc(szC * 4);
    ref_gemm(hA, hB, hR, M, N, K);
  }

  int8_t *dA, *dB;
  int32_t *dC;
  CHECK(mcMalloc((void **)&dA, szA));
  CHECK(mcMalloc((void **)&dB, szB));
  CHECK(mcMalloc((void **)&dC, szC * 4));
  CHECK(mcMemcpy(dA, hA, szA, mcMemcpyHostToDevice));
  CHECK(mcMemcpy(dB, hB, szB, mcMemcpyHostToDevice));

  const int iters = bench ? 200 : 0;
  Run p = run_one<false>(dA, dB, dC, hR, hC, M, N, K, iters);
  Run x = run_one<true>(dA, dB, dC, hR, hC, M, N, K, iters);

  printf("M=%d N=%d K=%d  grid %dx%d  LDS %d B\n", M, N, K, N / TILE_N,
         M / TILE_M, SMEM_BYTES);
  if (verify)
    printf("正确性   原始 %s (%zu/%zu 不符)   XOR %s (%zu/%zu 不符)\n",
           p.bad ? "FAIL" : "PASS", p.bad, szC, x.bad ? "FAIL" : "PASS", x.bad,
           szC);
  else
    printf("正确性   尺寸过大，跳过 CPU 参考\n");

  if (bench) {
    // 再跑一轮，这次先 XOR 后原始。两轮都跑是为了排掉「先跑的那个吃时钟爬坡」
    // 的顺序偏差 —— 两轮的加速比接近，结论才算数。
    Run x2 = run_one<true>(dA, dB, dC, NULL, hC, M, N, K, iters);
    Run p2 = run_one<false>(dA, dB, dC, NULL, hC, M, N, K, iters);
    const double flop = 2.0 * M * N * K;
    printf("        %12s %12s %10s\n", "原始 us", "XOR us", "加速");
    printf("第 1 轮 %12.3f %12.3f %9.3fx   (先原始)\n", p.us, x.us, p.us / x.us);
    printf("第 2 轮 %12.3f %12.3f %9.3fx   (先 XOR)\n", p2.us, x2.us,
           p2.us / x2.us);
    const double pb = p.us < p2.us ? p.us : p2.us;
    const double xb = x.us < x2.us ? x.us : x2.us;
    printf("取各自最快：原始 %.3f us / %.1f TOPS   XOR %.3f us / %.1f TOPS   "
           "加速 %.3fx\n",
           pb, flop / (pb * 1e-6) / 1e12, xb, flop / (xb * 1e-6) / 1e12,
           pb / xb);
  }

  free(hA);
  free(hB);
  free(hC);
  if (hR) free(hR);
  CHECK(mcFree(dA));
  CHECK(mcFree(dB));
  CHECK(mcFree(dC));
  return (p.bad || x.bad) ? 1 : 0;
}
