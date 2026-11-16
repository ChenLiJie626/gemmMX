// =============================================================================
// mma_swizzle_256.cu  ——  XOR swizzle 消 LDS bank 冲突，搬运走基线的异步通路
//
//   两个正交的开关，由同一份模板实例化，除了这两处以外逐字相同：
//     SW    = false / true    LDS 布局：原始 / XOR
//     ASYNC = false / true    搬运：普通指针拷贝 / 基线的 BSM 异步搬运
//   四个组合都验正确性、都计时，所以「异步值多少」和「XOR 值多少」分得开。
//
//   ---- 冲突在哪 ----
//   LDS 有 32 个 bank，每个 4 字节，一拍最多吐 32 * 4 = 128 字节。
//   一行 TILE_K = 256 字节，正好是 128 的整数倍 —— 每一行都从 bank 0 开始。
//   取片段时 16 条 lane 读 16 个不同的行、行内偏移却相同（同一个 q），
//   于是 64 条 lane 只用到 16 个 bank，每个排 16 个 dword = 16 拍，下限是 8 拍。
//
//   ---- XOR 怎么修 ----
//   一行 256B = 16 个 16B 块。让逻辑第 j 块住到第 (j ^ (r & 7)) 个槽位，
//   相邻 8 行就被错开到 8 个不同槽位，32 个 bank 全用上，8 拍，打到下限。
//
//   ---- 为什么 XOR 要提到 global 侧去做 ----
//   BSM 异步搬运（__builtin_mxc_ldg_b128_bsm_predicator）的 LDS 落点
//   **必须是按线程号连续的**：基线里写的就是
//       smem_A + rowA_s * kTileK + colA_s   ==   smem_A + tid * 16
//   这条约束下，「把第 j 块写到槽位 j^u」做不到 —— 落点不归我们挑。
//
//   但异或是**对合**的（a^b^b == a），所以下面两句话描述的是同一个布局：
//       (a) 把逻辑第 j 块，放到槽位 j ^ u        <- 改落点，异步搬运做不到
//       (b) 让槽位 s，去取逻辑第 s ^ u 块        <- 改源地址，照样能做
//   于是把异或挪到**全局侧的列号**上：LDS 落点保持连续（BSM 要的），
//   改的是「这条 lane 从 global 的哪一块取数」。读侧一个字都不用动。
//
//   代价：全局侧 16 条 lane 取的还是那 256 个连续字节，只是换了个人取 ——
//   **地址集合完全没变，合并访问的足迹也就没变**，只是 lane 顺序被打乱了。
//
//   编译：/opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_swizzle_256.cu -o mma_swizzle_256
//   运行：./mma_swizzle_256            四个组合各验一次正确性
//         ./mma_swizzle_256 banks      bank 直方图 + 布局自检（纯 host，不用 GPU）
//         ./mma_swizzle_256 bench      四个组合对照计时（可加 M N K）
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

// ---- 异步搬运（抄基线 kernel_test_v1.cu）----
#define arrive_gvmcnt(count) __builtin_mxc_arrive(64 + (count))
// MACA_ICMP_SLT = 40，见 __clang_maca_device_library_decls.h。
// 这里写成自己的常量，省得依赖 mctlass 头是否在包含路径上。
#define CMP_SLT 40

// ---- 读侧寻址：逻辑 (行 r, 行内字节 k) -> LDS 字节下标。k 必须 16B 对齐 ----
#define NOSWZ(r, k) ((r) * TILE_K + (k))
#define SWZ(r, k)                                                              \
  ((r) * TILE_K + (((((k) >> 4) ^ ((r) & SWZ_MASK)))<< 4) + ((k) & (CHUNK - 1)))
#define LDS_AT(SW, r, k) ((SW) ? SWZ(r, k) : NOSWZ(r, k))

// ---- 写侧：槽位 s 该去取逻辑第几块。对合，所以和上面是同一个布局 ----
#define SRC_CHUNK(SW, r, s) ((SW) ? ((s) ^ ((r) & SWZ_MASK)) : (s))

typedef __attribute__((ext_vector_type(4))) int i32x4;  // 累加器：4 个 int32
typedef __attribute__((ext_vector_type(4))) int i8x16;  // 16B = 4 个 MMA 片段

// -----------------------------------------------------------------------------
template <bool SW, bool ASYNC>
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

  // 搬运分工：16 个线程 × 16B = 256B = 一整行 TILE_K
  const int ld_row = tid / LDG_ROW_THREADS;   // 0..15  本轮搬第几行
  const int ld_slot = tid % LDG_ROW_THREADS;  // 0..15  LDS 落点的槽位，不能动
  // 这个槽位该装逻辑第几块。i 只会让行号加 16 的倍数，不影响 r & 7。
  const int ld_chunk = SRC_CHUNK(SW, ld_row, ld_slot);

  // 取片段用的 lane 坐标
  const int mn = lane % 16;
  const int q = lane / 16;

  for (int tk = 0; tk < K / TILE_K; ++tk) {
    // ---- 1. global -> LDS ----
    //   LDS 落点 r * TILE_K + ld_slot * 16 == i * 4096 + tid * 16，按线程号连续，
    //   这正是 BSM 的硬性要求；swizzle 体现在 ld_chunk（全局侧取第几块）。
#pragma unroll
    for (int i = 0; i < TILE_M / 16; ++i) {
      const int r = ld_row + i * 16;
      const int dst = r * TILE_K + ld_slot * CHUNK;
      const int koff = tk * TILE_K + ld_chunk * CHUNK;
      const int ga = block_row + r, gb = block_col + r;
      if (ASYNC) {
        __builtin_mxc_ldg_b128_bsm_predicator(
            As + dst, (void *)&A[(size_t)ga * K + koff],
            0, true, true, false, true, koff, K, CMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            Bs + dst, (void *)&B[(size_t)gb * K + koff],
            0, true, true, false, true, koff, K, CMP_SLT);
      } else {
        *(i8x16 *)&As[dst] = *(const i8x16 *)&A[(size_t)ga * K + koff];
        *(i8x16 *)&Bs[dst] = *(const i8x16 *)&B[(size_t)gb * K + koff];
      }
    }
    if (ASYNC) {
      arrive_gvmcnt(0);
      __builtin_mxc_barrier_inst();
    } else {
      __syncthreads();
    }

    // ---- 2. LDS -> 寄存器：冲突就出在这 32 条读上。这一段没有因为异步而改动 ----
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
    if (ASYNC)
      __builtin_mxc_barrier_inst();
    else
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
//  host 侧模型：把 kernel 里那几条寻址宏原样搬到 CPU 上数 bank、查布局。
//  不需要 GPU，也不需要 profiler —— bank 分布是地址的确定性函数。
// =============================================================================
static int lds_at(int sw, int r, int k) { return sw ? SWZ(r, k) : NOSWZ(r, k); }
static int src_chunk(int sw, int r, int s) { return SRC_CHUNK(sw, r, s); }

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

// 第 1 段的一条写指令：LDS 落点恒为 i*4096 + tid*16，两种布局完全一样
static void write_hist(int i, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int t = 0; t < WAVE; ++t)
    add_access(hist, (t / LDG_ROW_THREADS + i * 16) * TILE_K +
                         (t % LDG_ROW_THREADS) * CHUNK);
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

// 布局自检：写进去的是什么，读出来的就该是什么
//   写：LDS (行 r, 槽位 s) 装的是逻辑第 src_chunk(r, s) 块
//   读：想要逻辑第 j 块时，去 lds_at(r, j*16) 那个槽位取
// 两者必须对上，否则数值就是错的。
static int layout_roundtrip(int sw) {
  int bad = 0;
  for (int r = 0; r < TILE_M; ++r) {
    int slot_holds[TILE_K / CHUNK];
    for (int s = 0; s < TILE_K / CHUNK; ++s) slot_holds[s] = -1;
    for (int s = 0; s < TILE_K / CHUNK; ++s) {
      const int j = src_chunk(sw, r, s);
      if (j < 0 || j >= TILE_K / CHUNK) { ++bad; continue; }
      if (slot_holds[s] != -1) ++bad;
      slot_holds[s] = j;
    }
    // 每块都得有且只有一个家
    int seen[TILE_K / CHUNK];
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < TILE_K / CHUNK; ++s)
      if (slot_holds[s] >= 0 && seen[slot_holds[s]]++) ++bad;
    // 读回来必须是同一块
    for (int j = 0; j < TILE_K / CHUNK; ++j) {
      const int a = lds_at(sw, r, j * CHUNK);
      if (a < r * TILE_K || a >= (r + 1) * TILE_K) { ++bad; continue; }
      const int s = (a - r * TILE_K) / CHUNK;
      if (slot_holds[s] != j) ++bad;
    }
  }
  return bad;
}

// 全局侧足迹自检：同一行的 16 条 lane 取的还是那 256 个连续字节吗
static int global_footprint(int sw) {
  int bad = 0;
  for (int ld_row = 0; ld_row < 16; ++ld_row) {
    int seen[TILE_K / CHUNK];
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < LDG_ROW_THREADS; ++s) {
      const int j = src_chunk(sw, ld_row, s);
      if (j < 0 || j >= TILE_K / CHUNK) { ++bad; continue; }
      if (seen[j]++) ++bad;
    }
    for (int j = 0; j < TILE_K / CHUNK; ++j)
      if (seen[j] != 1) ++bad;
  }
  return bad;
}

static void bank_report(void) {
  const int floor_cycles = WAVE * CHUNK / TURN_B;  // 64*16/128 = 8
  int h[BANKS];

  printf("LDS：%d 个 bank × %d B = %d B/拍；一行 TILE_K = %d B = %d 轮\n", BANKS,
         BANK_B, TURN_B, TILE_K, TILE_K / TURN_B);
  printf("一条 wave 宽的 16B 访问要吐 %d × %d = %d B，下限 %d 拍\n\n", WAVE,
         CHUNK, WAVE * CHUNK, floor_cycles);

  printf("=== 第 2 段，一条 LDS 读指令 (base=0, s=0, c=0) 的 bank 深度 ===\n");
  printf("  %-6s", "bank");
  for (int b = 0; b < BANKS; ++b) printf("%3d", b);
  printf("\n");
  read_hist(0, 0, 0, 0, h);
  print_hist("原始", h);
  read_hist(1, 0, 0, 0, h);
  print_hist("XOR ", h);

  printf("\n=== 遍历全部 (base, s, c) 组合，取最坏 ===\n");
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
  {
    int worst = 0;
    for (int i = 0; i < TILE_M / 16; ++i) {
      write_hist(i, h);
      if (hist_max(h) > worst) worst = hist_max(h);
    }
    printf("  两种  写：最坏 %2d 拍（下限 %d）-> %s\n", worst, floor_cycles,
           worst == floor_cycles ? "打到下限" : "有冲突");
    printf("        （LDS 落点恒为 i*4096 + tid*16，两种布局完全相同 ——\n");
    printf("          这正是 BSM 异步搬运要求的「按线程号连续」）\n");
  }

  printf("\n=== 每个 k 窗口、每个 wave 的 LDS 读开销 ===\n");
  const int ninst = 2 * MMA_M * KCHUNK;  // A 16 条 + B 16 条
  printf("  原始 %d 条读 × 16 拍 = %4d 拍\n", ninst, ninst * 16);
  printf("  XOR  %d 条读 ×  8 拍 = %4d 拍   省掉 %d 拍\n", ninst, ninst * 8,
         ninst * 8);

  printf("\n=== 布局自检 ===\n");
  for (int sw = 0; sw < 2; ++sw)
    printf("  %-4s 写进去的块 <-> 读出来的块：%s\n", sw ? "XOR" : "原始",
           layout_roundtrip(sw) ? "对不上！" : "全部对上（128 行 × 16 块）");
  for (int sw = 0; sw < 2; ++sw)
    printf("  %-4s 全局侧 16 条 lane 的足迹：%s\n", sw ? "XOR" : "原始",
           global_footprint(sw) ? "变了！"
                                : "仍是那 256 个连续字节，只换了取数的人");
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

template <bool SW, bool ASYNC>
static Run run_one(const int8_t *dA, const int8_t *dB, int32_t *dC,
                   const int32_t *hR, int32_t *hC, int M, int N, int K,
                   int iters) {
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  Run r;
  r.us = 0;
  r.bad = 0;

  CHECK(mcMemset(dC, 0, (size_t)M * N * 4));
  mma_256<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
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
      mma_256<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_256<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
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

static const char *kName[4] = {"同步搬运 + 原始布局", "同步搬运 + XOR 布局",
                               "异步搬运 + 原始布局", "异步搬运 + XOR 布局"};

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
  Run r[4], r2[4];
  r[0] = run_one<false, false>(dA, dB, dC, hR, hC, M, N, K, iters);
  r[1] = run_one<true, false>(dA, dB, dC, hR, hC, M, N, K, iters);
  r[2] = run_one<false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
  r[3] = run_one<true, true>(dA, dB, dC, hR, hC, M, N, K, iters);

  printf("M=%d N=%d K=%d  grid %dx%d  LDS %d B\n", M, N, K, N / TILE_N,
         M / TILE_M, SMEM_BYTES);
  if (verify) {
    printf("\n=== 正确性（对 CPU 参考）===\n");
    for (int i = 0; i < 4; ++i)
      printf("  %-24s %s  (%zu/%zu 不符)\n", kName[i],
             r[i].bad ? "FAIL" : "PASS", r[i].bad, szC);
  } else {
    printf("正确性   尺寸过大，跳过 CPU 参考\n");
  }

  if (bench) {
    // 倒着再跑一轮，排掉「先跑的那个吃时钟爬坡」的顺序偏差
    r2[3] = run_one<true, true>(dA, dB, dC, NULL, hC, M, N, K, iters);
    r2[2] = run_one<false, true>(dA, dB, dC, NULL, hC, M, N, K, iters);
    r2[1] = run_one<true, false>(dA, dB, dC, NULL, hC, M, N, K, iters);
    r2[0] = run_one<false, false>(dA, dB, dC, NULL, hC, M, N, K, iters);

    const double flop = 2.0 * M * N * K;
    double best[4];
    for (int i = 0; i < 4; ++i) best[i] = r[i].us < r2[i].us ? r[i].us : r2[i].us;

    printf("\n=== 计时（两轮，第二轮顺序颠倒；下表取各自最快）===\n");
    printf("  %-24s %10s %10s %10s %10s\n", "组合", "第1轮us", "第2轮us",
           "取快us", "TOPS");
    for (int i = 0; i < 4; ++i)
      printf("  %-24s %10.3f %10.3f %10.3f %10.1f\n", kName[i], r[i].us,
             r2[i].us, best[i], flop / (best[i] * 1e-6) / 1e12);

    printf("\n=== 拆开看每一项值多少（相对「同步 + 原始」）===\n");
    printf("  只上异步搬运        %.3fx\n", best[0] / best[2]);
    printf("  只上 XOR 布局       %.3fx\n", best[0] / best[1]);
    printf("  两个都上            %.3fx\n", best[0] / best[3]);
    printf("  异步之上再加 XOR    %.3fx\n", best[2] / best[3]);
  }

  free(hA);
  free(hB);
  free(hC);
  if (hR) free(hR);
  CHECK(mcFree(dA));
  CHECK(mcFree(dB));
  CHECK(mcFree(dC));
  for (int i = 0; i < 4; ++i)
    if (r[i].bad) return 1;
  return 0;
}
