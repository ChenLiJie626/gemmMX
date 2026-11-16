// =============================================================================
// mma_swizzle_c600.cu  ——  mma_swizzle_256.cu 的 C600(xcore1500)移植版
//
//   一份源码两个目标：
//     -DMX_ARCH=500   C500 / xcore1000 / MMA K=16   <- 与 mma_swizzle_256.cu 等价
//     -DMX_ARCH=600   C600 / xcore1500 / MMA K=32   <- 本文件的目的
//
//   编译（C600）：
//     /opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 -std=c++17 \
//         --offload-arch=xcore1500 -DMX_ARCH=600 \
//         mma_swizzle_c600.cu -o mma_swizzle_c600
//   编译（C500，用来回归这份源码没改坏 K=16 的路径）：
//     mxcc -x maca -O3 -std=c++17 -offload-arch native -DMX_ARCH=500 ...
//
//   运行：
//     ./mma_swizzle_c600            四个组合各验一次正确性
//     ./mma_swizzle_c600 arch       编译期假设 vs 真机属性逐条核对（上机第一件事）
//     ./mma_swizzle_c600 banks      bank 直方图 + 布局自检（纯 host，不用 GPU）
//     ./mma_swizzle_c600 bench      四个组合对照计时（可加 M N K）
//
// -----------------------------------------------------------------------------
//  C600 和 C500 的差别，以及这份源码为此改了什么
// -----------------------------------------------------------------------------
//  【已知】C600 是 XCORE 1.5 架构（公开报道），C500 是 xcore1000（真机实测）。
//  【已知】SDK 的 __clang_maca_mma_functions.h 里有 __builtin_mxc_mma_16x16x32i8，
//          它的 target feature 门是 xcore1500|1520|1600|2000。C500 上
//          -offload-arch=xcore1500 能编过，一跑就是
//          ErrorNoBinaryForGpu: Couldn't find xcore1000 binary。
//  【推论】C600 = xcore1500 系，于是 **K=32 的 int8 MMA 在 C600 上可用**。
//          这是两颗芯片之间对本算子唯一的结构性差别，也是本文件的全部改动来源。
//
//  一条 K=32 的 MMA 顶两条 K=16：同样的 LDS 读、同样的搬运、同样的 acc，
//  MMA 指令数砍半 —— 换句话说，非 MMA 指令在发射槽里的占比翻倍地刺眼，
//  所以消 bank 冲突这件事在 C600 上比在 C500 上更值钱，而不是更不值钱。
//
//  ---- 为什么读侧一个字都不用改 ----
//  C500：一条 16B 的 LDS 读 = 4 个 4B 片段 = 喂 4 条 K=16 的 MMA。
//  C600：同一条 16B 读 = 2 个 8B 片段 = 喂 2 条 K=32 的 MMA。
//  地址、步长、bank 行为**完全相同**，只是把这 16 个字节切成 2 份而不是 4 份。
//  于是 XOR swizzle（下面那两条宏）在 C600 上原样成立，mask 还是 7。
//
//  ---- 为什么这样切 k 是对的，哪怕 K=32 的片段布局还没核过 ----
//  硬件只保证一件事：a 的第 i 个字节槽和 b 的第 i 个字节槽配到同一个 k。
//  它认为那是第几个 k，我们不关心 —— 只要
//    (1) 同一条 lane 给 A 和给 B 的 8 个字节取自同一组逻辑 k（代码对称，成立）；
//    (2) 4 个 lane 组的 32 个 k 互不相同（q*8 错开，成立）；
//    (3) 8 条指令 (c, w) 的 k 集合两两不交、并起来是整个窗口（见下面的推导，成立）；
//  结果就是对的。这正是教程里「k 的置换不影响结果」那条结论的直接复用，
//  所以**不必等到拿到 K=32 的片段布局文档才能写这个 kernel**。
//
//    指令 (c, w) 从 lane q 取 LDS 字节 64c + 16q + 8w .. +7
//    它的 k 集合 = { 64c + 16q + 8w + j : q=0..3, j=0..7 }   共 32 个
//    c=0..3, w=0..1 一共 8 条指令 × 32 = 256 = 一个 k 窗口，不重不漏。
//
//  ---- 还没核实、第一次上真机要查的东西 ----
//   A. arch 串到底是 xcore1500 还是 1520/1600。`mxcc --offload-arch=help`
//      或 mx-smi 能给。本文件只在注释里写，不硬编码。
//   B. mma_16x16x32i8 的真实签名。本文件按「a/b 各是一个 64 位打包」写成 long；
//      若 SDK 写的是 int2，加 -DMX_FRAG64_AS_V2I32 即可。核对命令：
//        grep -n 'mma_16x16x32i8' \
//          /opt/maca/mxgpu_llvm/lib/clang/*/include/__clang_maca_mma_functions.h
//   C. 累加器布局。输出块还是 16x16，所以极可能和 K=16 逐字相同
//      （m0=(lane>>4)*4, n=lane&15）。不同的话正确性检查会立刻 FAIL，
//      改第 4 段那两行即可。
//   D. 每 block 的 LDS 上限、CU 数、wave 宽度 —— `./mma_swizzle_c600 arch`
//      会把编译期假设和真机属性逐条列出来对照，不一致的行标 MISMATCH。
//   E. BSM 异步搬运 builtin 在 xcore1500 上是否同名同语义。不同名就
//      -DMX_HAS_BSM=0，退回同步拷贝，XOR 侧不受影响。
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

// -----------------------------------------------------------------------------
// 0. 目标芯片
// -----------------------------------------------------------------------------
// __MACA_ARCH__ 这个预定义宏在本项目里**没有实测过**（只在旧注释里见过），
// 所以只拿它当默认值的来源；真正决定走哪条路的是 -DMX_ARCH。
#ifndef MX_ARCH
#if defined(__MACA_ARCH__) && __MACA_ARCH__ >= 1500
#define MX_ARCH 600
#else
#define MX_ARCH 500
#endif
#endif

#ifndef MX_MMA_K
#if MX_ARCH >= 600
#define MX_MMA_K 32   // C600：一条 MMA 吃 32 个 k
#else
#define MX_MMA_K 16   // C500：一条 MMA 吃 16 个 k
#endif
#endif

#ifndef MX_HAS_BSM
#define MX_HAS_BSM 1  // 0 = 没有 BSM 异步搬运，ASYNC 组合退化成同步拷贝
#endif

#if MX_ARCH >= 600
#define MX_ARCH_NAME "C600 / xcore1500 系"
#else
#define MX_ARCH_NAME "C500 / xcore1000"
#endif

// -----------------------------------------------------------------------------
// 1. 分块参数（全部可 -D 覆盖）
//    默认值 = C500 上实测过的那一组，C600 上先用它建立正确性基线，
//    再按 `arch` 报告里的算术强度那几行往上调。
// -----------------------------------------------------------------------------
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 256
#endif
#ifndef WAVE
#define WAVE 64          // 一个 wave 几条 lane
#endif
#ifndef WAVE_M
#define WAVE_M 2         // wave 在 M 方向排几个
#endif
#ifndef WAVE_N
#define WAVE_N 2
#endif

#define THREADS (WAVE * WAVE_M * WAVE_N)
#define WM (TILE_M / WAVE_M)
#define WN (TILE_N / WAVE_N)
#define MMA_M (WM / 16)
#define MMA_N (WN / 16)
#define SMEM_BYTES ((TILE_M + TILE_N) * TILE_K)

// ---- bank 参数（硬件常量；C600 上由 `arch` 模式提示复核）----
#define BANKS 32                          // LDS bank 数
#define BANK_B 4                          // 每个 bank 的宽度
#define CHUNK 16                          // 一次 LDS 访问的宽度（b128）
#define TURN_B (BANKS * BANK_B)           // 128  一拍吐得出的字节数
#define CHUNKS_PER_TURN (TURN_B / CHUNK)  // 8    一拍装得下几个 16B 块
#define CHUNKS_PER_ROW (TILE_K / CHUNK)   // 16   一行有几个 16B 块
#define MX_MIN(a, b) ((a) < (b) ? (a) : (b))
// 异或的周期：最多 8（一拍的块数），且不能越过行边界
#define SWZ_MASK (MX_MIN(CHUNKS_PER_TURN, CHUNKS_PER_ROW) - 1)

// ---- 读侧的 k 分组 ----
#define LANE_GROUPS (WAVE / 16)                  // 4 个 lane 组（q）
#define KGROUP (LANE_GROUPS * CHUNK)             // 64B：一个 c 步跨多少字节
#define KCHUNK (TILE_K / KGROUP)                 // 4：c 的取值个数
// 一个片段的字节数：一条 MMA 要 16 行 × K 个 int8 = 16K 字节，64 条 lane 分，
// 每条 lane 16K/64 = K/4 字节。K=16 -> 4B，K=32 -> 8B。
#define MX_FRAG_B (MX_MMA_K / 4)
#define MMA_PER_CHUNK (CHUNK / MX_FRAG_B)        // 一条 16B 读喂几条 MMA
                                                 //   K=16 -> 4，K=32 -> 2

// ---- 搬运分工 ----
#define LDG_ROW_THREADS CHUNKS_PER_ROW           // 搬一整行要几条 lane
#define LDG_ROWS (THREADS / LDG_ROW_THREADS)     // 一趟搬几行
#define LDG_PASSES_A (TILE_M / LDG_ROWS)
#define LDG_PASSES_B (TILE_N / LDG_ROWS)

// ---- 异步搬运 ----
#define arrive_gvmcnt(count) __builtin_mxc_arrive(64 + (count))
// MACA_ICMP_SLT = 40，见 __clang_maca_device_library_decls.h
#define CMP_SLT 40

// ---- 读侧寻址：逻辑 (行 r, 行内字节 k) -> LDS 字节下标。k 必须 16B 对齐 ----
#define NOSWZ(r, k) ((r) * TILE_K + (k))
#define SWZ(r, k)                                                              \
  ((r) * TILE_K + (((((k) >> 4) ^ ((r) & SWZ_MASK))) << 4) + ((k) & (CHUNK - 1)))
#define LDS_AT(SW, r, k) ((SW) ? SWZ(r, k) : NOSWZ(r, k))

// ---- 写侧：槽位 s 该去取逻辑第几块。对合，所以和上面是同一个布局 ----
#define SRC_CHUNK(SW, r, s) ((SW) ? ((s) ^ ((r) & SWZ_MASK)) : (s))

// -----------------------------------------------------------------------------
// 2. MMA 原语：C500 / C600 唯一的真差别就在这几行
// -----------------------------------------------------------------------------
typedef __attribute__((ext_vector_type(4))) int i32x4;  // 累加器：4 个 int32

// mx_frag 是**片段的存放类型**，必须是标量：下面要拿它做 ext_vector_type 的
// 元素类型，而 clang 不允许「向量的向量」。builtin 真实操作数类型若不是它，
// 在调用处转一下（mx_as_op），-O3 下这就是个寄存器改名，不生成指令。
#if MX_MMA_K == 32
typedef long mx_frag;  // 8B = 8 个 int8
#ifdef MX_FRAG64_AS_V2I32
typedef __attribute__((ext_vector_type(2))) int mx_op;  // 备选签名：int2
static __device__ __forceinline__ mx_op mx_as_op(mx_frag v) {
  mx_op r;
  __builtin_memcpy(&r, &v, sizeof(r));
  return r;
}
#else
typedef mx_frag mx_op;
static __device__ __forceinline__ mx_op mx_as_op(mx_frag v) { return v; }
#endif
#define MX_MMA(a, b, c)                                                        \
  __builtin_mxc_mma_16x16x32i8(mx_as_op(a), mx_as_op(b), (c))
#define MX_MMA_NAME "__builtin_mxc_mma_16x16x32i8"
#elif MX_MMA_K == 16
typedef int mx_frag;  // 4B = 4 个 int8
#define MX_MMA(a, b, c) __builtin_mxc_mma_16x16x16i8((a), (b), (c))
#define MX_MMA_NAME "__builtin_mxc_mma_16x16x16i8"
#else
#error "MX_MMA_K 只支持 16 或 32"
#endif

// 一次 16B 的 LDS / global 访问单元，按片段切好
typedef __attribute__((ext_vector_type(MMA_PER_CHUNK))) mx_frag mx_tile;

// -----------------------------------------------------------------------------
// 3. 编译期的假设墙
//    下面每一条都是这份 kernel 正确性所依赖的东西。C600 上要是哪条不成立，
//    **编译就该直接失败**，而不是算出一堆错数字。
// -----------------------------------------------------------------------------
static_assert(WAVE == 64,
              "MMA 的 lane 映射（行=lane%16，k 组=lane/16）写死了 wave=64；"
              "C600 若不是 64，第 2/4 段都要重推");
static_assert(sizeof(mx_frag) == MX_FRAG_B,
              "片段字节数必须是 K/4（16 行 × K 个 int8 分给 64 条 lane）");
static_assert(LANE_GROUPS * MX_FRAG_B == MX_MMA_K,
              "4 个 lane 组各出一个片段，合起来必须正好是一条 MMA 的 K");
static_assert(sizeof(mx_tile) == CHUNK, "16B 读单元没对上");
static_assert(CHUNK % MX_FRAG_B == 0, "16B 必须能整切成整数个片段");
static_assert(MMA_PER_CHUNK >= 1, "一条 16B 读至少要供一条 MMA");
static_assert((CHUNKS_PER_ROW & (CHUNKS_PER_ROW - 1)) == 0,
              "一行的块数必须是 2 的幂，否则异或会跑出行外");
static_assert((CHUNKS_PER_TURN & (CHUNKS_PER_TURN - 1)) == 0,
              "一拍的块数必须是 2 的幂");
static_assert(TILE_K % KGROUP == 0, "TILE_K 要能被 64B 的 c 步整除");
static_assert(TILE_K % MX_MMA_K == 0, "k 窗口要能被一条 MMA 的 K 整除");
static_assert(KCHUNK * MMA_PER_CHUNK * MX_MMA_K == TILE_K,
              "(c, w) 两层循环必须正好铺满一个 k 窗口，不重不漏");
static_assert(WM % 16 == 0 && WN % 16 == 0, "wave tile 要是 16 的倍数");
static_assert(THREADS % LDG_ROW_THREADS == 0, "线程数要能整除成整行");
static_assert(TILE_M % LDG_ROWS == 0 && TILE_N % LDG_ROWS == 0,
              "tile 的行数要能被一趟搬的行数整除");
static_assert(SMEM_BYTES <= 65536,
              "LDS 超 64KB。C600 的上限还没核过，真机上看 `arch` 报告那一行");

// -----------------------------------------------------------------------------
// 4. 搬运：global -> LDS
//    LDS 落点恒为「按线程号连续」（BSM 的硬性要求），swizzle 全在全局侧的
//    列号上做。异或是对合，所以这和「把第 j 块写到槽位 j^u」是同一个布局。
// -----------------------------------------------------------------------------
template <bool SW, bool ASYNC, int PASSES>
__device__ __forceinline__ void load_tile(int8_t *dst, const int8_t *src,
                                          int grow0, int K, int tk, int ld_row,
                                          int ld_slot) {
#pragma unroll
  for (int i = 0; i < PASSES; ++i) {
    const int r = ld_row + i * LDG_ROWS;
    const int j = SRC_CHUNK(SW, r, ld_slot);          // 全局侧取第几块
    const int off = r * TILE_K + ld_slot * CHUNK;     // LDS 落点，连续，不能动
    const int koff = tk * TILE_K + j * CHUNK;
    const int8_t *p = src + (size_t)(grow0 + r) * K + koff;
#if MX_HAS_BSM
    if (ASYNC) {
      __builtin_mxc_ldg_b128_bsm_predicator(dst + off, (void *)p, 0, true, true,
                                            false, true, koff, K, CMP_SLT);
      continue;
    }
#endif
    *(mx_tile *)(dst + off) = *(const mx_tile *)p;
  }
}

// -----------------------------------------------------------------------------
// 5. kernel
// -----------------------------------------------------------------------------
template <bool SW, bool ASYNC>
__global__ __launch_bounds__(THREADS) void mma_tile(
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

  const int ld_row = tid / LDG_ROW_THREADS;   // 本趟搬第几行
  const int ld_slot = tid % LDG_ROW_THREADS;  // LDS 落点的槽位，不能动

  const int mn = lane % 16;                   // 行 / 列
  const int q = lane / 16;                    // k 组

  const bool async = ASYNC && (MX_HAS_BSM != 0);
  (void)async;  // MX_HAS_BSM=0 时用不到

  for (int tk = 0; tk < K / TILE_K; ++tk) {
    // ---- 1. global -> LDS ----
    load_tile<SW, ASYNC, LDG_PASSES_A>(As, A, block_row, K, tk, ld_row, ld_slot);
    load_tile<SW, ASYNC, LDG_PASSES_B>(Bs, B, block_col, K, tk, ld_row, ld_slot);
#if MX_HAS_BSM
    if (async) {
      arrive_gvmcnt(0);
      __builtin_mxc_barrier_inst();
    } else
#endif
      __syncthreads();

    // ---- 2. LDS -> 寄存器：bank 冲突就出在这些读上 ----
    //   一条读 16B；K=16 时切 4 份、K=32 时切 2 份，地址与 bank 行为不变。
    mx_tile af[MMA_M][KCHUNK], bf[MMA_N][KCHUNK];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * CHUNK + c * KGROUP;
        const int ra = wave_m * WM + s * 16 + mn;
        af[s][c] = *(const mx_tile *)&As[LDS_AT(SW, ra, k_off)];
      }
#pragma unroll
    for (int s = 0; s < MMA_N; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * CHUNK + c * KGROUP;
        const int rb = wave_n * WN + s * 16 + mn;
        bf[s][c] = *(const mx_tile *)&Bs[LDS_AT(SW, rb, k_off)];
      }

    // ---- 3. MMA ----
    //   C500: KCHUNK(4) * MMA_PER_CHUNK(4) = 16 条 / (m,n)
    //   C600: KCHUNK(4) * MMA_PER_CHUNK(2) =  8 条 / (m,n)   ← 砍半
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
      for (int n = 0; n < MMA_N; ++n)
#pragma unroll
        for (int c = 0; c < KCHUNK; ++c)
#pragma unroll
          for (int w = 0; w < MMA_PER_CHUNK; ++w)
            acc[m][n] = MX_MMA(af[m][c][w], bf[n][c][w], acc[m][n]);

#if MX_HAS_BSM
    if (async)
      __builtin_mxc_barrier_inst();
    else
#endif
      __syncthreads();
  }

  // ---- 4. 写回：acc 布局假设与 K=16 相同（输出块仍是 16x16）----
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

static void add_access(int *hist, int byte_addr) {
  for (int d = 0; d < CHUNK / BANK_B; ++d)
    hist[(byte_addr / BANK_B + d) % BANKS]++;
}

// 第 2 段的一条读指令：64 条 lane，(base, s, c) 固定
static void read_hist(int sw, int base, int s, int c, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int lane = 0; lane < WAVE; ++lane) {
    const int r = base + s * 16 + (lane % 16);
    const int k = (lane / 16) * CHUNK + c * KGROUP;
    add_access(hist, lds_at(sw, r, k));
  }
}

// 第 1 段的一条写指令：LDS 落点恒为按线程号连续，两种布局完全一样
static void write_hist(int i, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int t = 0; t < WAVE; ++t)
    add_access(hist, (t / LDG_ROW_THREADS + i * LDG_ROWS) * TILE_K +
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
static int layout_roundtrip(int sw) {
  int bad = 0;
  for (int r = 0; r < TILE_M; ++r) {
    int slot_holds[CHUNKS_PER_ROW];
    for (int s = 0; s < CHUNKS_PER_ROW; ++s) slot_holds[s] = -1;
    for (int s = 0; s < CHUNKS_PER_ROW; ++s) {
      const int j = src_chunk(sw, r, s);
      if (j < 0 || j >= CHUNKS_PER_ROW) { ++bad; continue; }
      if (slot_holds[s] != -1) ++bad;
      slot_holds[s] = j;
    }
    int seen[CHUNKS_PER_ROW];
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < CHUNKS_PER_ROW; ++s)
      if (slot_holds[s] >= 0 && seen[slot_holds[s]]++) ++bad;
    for (int j = 0; j < CHUNKS_PER_ROW; ++j) {
      const int a = lds_at(sw, r, j * CHUNK);
      if (a < r * TILE_K || a >= (r + 1) * TILE_K) { ++bad; continue; }
      const int s = (a - r * TILE_K) / CHUNK;
      if (slot_holds[s] != j) ++bad;
    }
  }
  return bad;
}

// 全局侧足迹自检：同一行的若干条 lane 取的还是那一段连续字节吗
static int global_footprint(int sw) {
  int bad = 0;
  for (int ld_row = 0; ld_row < LDG_ROWS; ++ld_row) {
    int seen[CHUNKS_PER_ROW];
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < LDG_ROW_THREADS; ++s) {
      const int j = src_chunk(sw, ld_row, s);
      if (j < 0 || j >= CHUNKS_PER_ROW) { ++bad; continue; }
      if (seen[j]++) ++bad;
    }
    for (int j = 0; j < CHUNKS_PER_ROW; ++j)
      if (seen[j] != 1) ++bad;
  }
  return bad;
}

// k 覆盖自检：(c, w) 两层循环铺出来的 k 集合，不重不漏才算对
//   指令 (c, w) 的 k 集合 = { c*KGROUP + q*CHUNK + w*MX_FRAG_B + j }
//                            q = 0..LANE_GROUPS-1, j = 0..MX_FRAG_B-1
static int kcover(void) {
  const int FB = MX_FRAG_B;  // 一个片段几个字节 = 几个 k
  int cnt[TILE_K];
  memset(cnt, 0, sizeof(cnt));
  for (int c = 0; c < KCHUNK; ++c)
    for (int w = 0; w < MMA_PER_CHUNK; ++w)
      for (int q = 0; q < LANE_GROUPS; ++q)
        for (int j = 0; j < FB; ++j) {
          const int k = c * KGROUP + q * CHUNK + w * FB + j;
          if (k < 0 || k >= TILE_K) return -1;
          cnt[k]++;
        }
  int bad = 0;
  for (int k = 0; k < TILE_K; ++k)
    if (cnt[k] != 1) ++bad;
  return bad;
}

static void config_report(void) {
  printf("目标        %s   MMA K=%d   %s\n", MX_ARCH_NAME, MX_MMA_K,
         MX_MMA_NAME);
  printf("tile        M=%d N=%d K=%d   线程 %d = %d wave 排 %dx%d\n", TILE_M,
         TILE_N, TILE_K, THREADS, THREADS / WAVE, WAVE_M, WAVE_N);
  printf("每 wave     %dx%d，MMA tile %dx%d，acc %d 个 i32x4 = %d 个 VGPR\n", WM,
         WN, MMA_M, MMA_N, MMA_M * MMA_N, MMA_M * MMA_N * 4);
  printf("LDS         %d B（A %d + B %d）\n", SMEM_BYTES, TILE_M * TILE_K,
         TILE_N * TILE_K);
  printf("片段        %d B -> 一条 16B 读供 %d 条 MMA\n", (int)sizeof(mx_frag),
         MMA_PER_CHUNK);
  printf("异步搬运    %s\n", MX_HAS_BSM ? "BSM b128（编译期认为可用）" : "关闭");
}

static void bank_report(void) {
  const int floor_cycles = WAVE * CHUNK / TURN_B;
  int h[BANKS];

  config_report();
  printf("\nLDS：%d 个 bank × %d B = %d B/拍；一行 TILE_K = %d B = %d 轮\n",
         BANKS, BANK_B, TURN_B, TILE_K, TILE_K / TURN_B);
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

  printf("\n=== 遍历全部 (base, s, c) 组合，取最坏 ===\n");
  int worst_read[2] = {0, 0};
  for (int sw = 0; sw < 2; ++sw) {
    for (int wmi = 0; wmi < WAVE_M; ++wmi)
      for (int s = 0; s < MMA_M; ++s)
        for (int c = 0; c < KCHUNK; ++c) {
          read_hist(sw, wmi * WM, s, c, h);
          if (hist_max(h) > worst_read[sw]) worst_read[sw] = hist_max(h);
        }
    printf("  %-4s 读：最坏 %2d 拍（下限 %d）-> %s\n", sw ? "XOR" : "原始",
           worst_read[sw], floor_cycles,
           worst_read[sw] == floor_cycles ? "打到下限" : "有冲突");
  }
  {
    int worst = 0;
    for (int i = 0; i < LDG_PASSES_A; ++i) {
      write_hist(i, h);
      if (hist_max(h) > worst) worst = hist_max(h);
    }
    printf("  两种  写：最坏 %2d 拍（下限 %d）-> %s\n", worst, floor_cycles,
           worst == floor_cycles ? "打到下限" : "有冲突");
  }

  printf("\n=== 每个 k 窗口、每个 wave 的指令账 ===\n");
  const int nread = (MMA_M + MMA_N) * KCHUNK;
  const int nmma = MMA_M * MMA_N * KCHUNK * MMA_PER_CHUNK;
  printf("  LDS 读 %3d 条：原始 × %d 拍 = %4d 拍，XOR × %d 拍 = %4d 拍\n", nread,
         worst_read[0], nread * worst_read[0], worst_read[1],
         nread * worst_read[1]);
  printf("  MMA    %3d 条（K=%d）\n", nmma, MX_MMA_K);
  printf("  读 : MMA = 1 : %.1f   —— MMA K 翻倍会把这个比值砍半，\n",
         (double)nmma / nread);
  printf("                        所以消冲突在 C600 上更值钱，不是更不值钱。\n");

  printf("\n=== 自检 ===\n");
  for (int sw = 0; sw < 2; ++sw)
    printf("  %-4s 写进去的块 <-> 读出来的块：%s\n", sw ? "XOR" : "原始",
           layout_roundtrip(sw) ? "对不上！" : "全部对上");
  for (int sw = 0; sw < 2; ++sw)
    printf("  %-4s 全局侧一行内的足迹：%s\n", sw ? "XOR" : "原始",
           global_footprint(sw) ? "变了！" : "仍是那一段连续字节，只换了取数的人");
  const int kb = kcover();
  printf("  (c, w) 两层循环铺出的 %d 个 k：%s\n", TILE_K,
         kb == 0 ? "不重不漏" : "有重或有漏！");
}

// =============================================================================
//  arch 模式：编译期假设 vs 真机属性。C600 上机第一件事就跑这个。
// =============================================================================
template <bool SW, bool ASYNC>
static int occupancy_of(void) {
  int blocks = -1;
  mcError_t e = mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, (const void *)mma_tile<SW, ASYNC>, THREADS, SMEM_BYTES);
  return (e == mcSuccess) ? blocks : -1;
}

static void prow(const char *what, const char *assumed, const char *actual,
                int ok) {
  printf("  %-22s %-26s %-26s %s\n", what, assumed, actual,
         ok < 0 ? "?" : (ok ? "OK" : "MISMATCH"));
}

static void arch_report(void) {
  config_report();

  int dev = 0;
  mcDeviceProp_t p;
  CHECK(mcGetDevice(&dev));
  CHECK(mcGetDeviceProperties(&p, dev));

  char a[64], b[64];
  printf("\n=== 编译期假设 vs 真机 ===\n");
  printf("  %-22s %-26s %-26s %s\n", "项", "编译期认为", "真机", "");
  prow("设备名", MX_ARCH_NAME, p.name, -1);

  snprintf(a, sizeof(a), "<= %d B", SMEM_BYTES);
  snprintf(b, sizeof(b), "%zu B", (size_t)p.sharedMemPerBlock);
  prow("LDS / block", a, b, SMEM_BYTES <= (int)p.sharedMemPerBlock);

  snprintf(a, sizeof(a), "（不假设）");
  snprintf(b, sizeof(b), "%d 个", p.multiProcessorCount);
  prow("CU 数", a, b, -1);

  snprintf(b, sizeof(b), "%d MHz", p.clockRate / 1000);
  prow("时钟", a, b, -1);

  snprintf(a, sizeof(a), "%d 条 lane", WAVE);
  prow("wave 宽度", a, "device prop 里没有，见下注", -1);

  snprintf(a, sizeof(a), "%d 个 × %d B", BANKS, BANK_B);
  prow("LDS bank", a, "查不到，只能由时序反推", -1);

  const int occ = occupancy_of<true, true>();
  snprintf(b, sizeof(b), "%d 个 block/CU", occ);
  prow("驻留度", "（看实测）", b, occ > 0);

  printf("\n  wave 宽度没有可查的属性，用一条 kernel 反推：\n");
  printf("    若 wave != 64，第 2 段的 (lane%%16, lane/16) 和第 4 段的 acc\n");
  printf("    布局都会错，正确性检查会 FAIL —— 跑 ./mma_swizzle_c600 就知道。\n");

  printf("\n=== 算术强度（决定 tile 该不该加大）===\n");
  const double opb = 2.0 * TILE_M * TILE_N / (TILE_M + TILE_N);
  printf("  本 tile：每从 global 读 1 字节做 %.0f 次 int8 乘加\n", opb);
  printf("  要想不撞带宽墙，这个数得 >= 峰值算力 / 带宽：\n");
  printf("    C600 公开数（未核实，仅作参考）2000 TOPS / 3.35 TB/s = %.0f\n",
         2000e12 / 3.35e12);
  const double need = 2000e12 / 3.35e12;
  printf("  %.0f %s %.0f，%s\n", opb, opb < need ? "<" : ">=", need,
         opb < need ? "缺口要靠 L2 复用补。补不上就把 tile 加大：" : "够了。");
  printf("    -DTILE_M=256 -DWAVE_M=4 -DWAVE_N=2 -DTILE_K=128\n");
  printf("    -> 256x128 tile，8 wave，wave tile 还是 64x64，\n");
  printf("       LDS %d B，acc 还是 16 个 i32x4 = 64 个 VGPR，每字节 %.0f 次乘加\n",
         (256 + 128) * 128, 2.0 * 256 * 128 / (256 + 128));
  printf("  这一组在 C500 的 fp16 算子上是扫参最优点，int8 / C600 上没验过。\n");
  printf("  （host 侧的 bank 模型已经确认这组参数下 XOR 仍然打到下限：\n");
  printf("    用 -DTILE_M=256 -DTILE_K=128 -DWAVE_M=4 ... banks 自己看）\n");
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
  mma_tile<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
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
      mma_tile<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_tile<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
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
  if (argc > 1 && strcmp(argv[1], "arch") == 0) {
    arch_report();
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
    fprintf(stderr, "M 需为 %d 的倍数，N 需为 %d 的倍数，K 需为 %d 的倍数\n",
            TILE_M, TILE_N, TILE_K);
    return 1;
  }

  const size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
  int8_t *hA = (int8_t *)malloc(szA), *hB = (int8_t *)malloc(szB);
  int32_t *hC = (int32_t *)malloc(szC * 4), *hR = NULL;
  srand(1234);
  for (size_t i = 0; i < szA; ++i) hA[i] = (int8_t)((rand() % 15) - 7);
  for (size_t i = 0; i < szB; ++i) hB[i] = (int8_t)((rand() % 15) - 7);

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

  printf("%s   MMA K=%d\n", MX_ARCH_NAME, MX_MMA_K);
  printf("M=%d N=%d K=%d  grid %dx%d  LDS %d B\n", M, N, K, N / TILE_N,
         M / TILE_M, SMEM_BYTES);
  if (verify) {
    printf("\n=== 正确性（对 CPU 参考）===\n");
    for (int i = 0; i < 4; ++i)
      printf("  %-24s %s  (%zu/%zu 不符)\n", kName[i],
             r[i].bad ? "FAIL" : "PASS", r[i].bad, szC);
    printf("\n  全 FAIL 且四个一模一样 -> 先怀疑 acc 布局或 wave 宽度，\n");
    printf("  不是 swizzle（swizzle 错只会让带 XOR 的两行 FAIL）。\n");
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
