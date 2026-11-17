// =============================================================================
// mma_dbuf_256.cu  ——  切 K 换出双缓冲：让搬运真正藏进 MMA 后面
//
//   接着 mma_swizzle_256.cu / mma_pipeline_256.cu 往下做。前面两步的结论：
//     * XOR 布局把 LDS 读打到下限，4096^3 拿到 1.85x；
//     * 异步搬运单独上只值 1.1% —— 发出去立刻 arrive_gvmcnt(0)，中间没活干；
//     * TILE_K=256 时一个缓冲就 2*128*256 = 65536 B，顶满 64KB，**双缓冲放不下**。
//
//   ---- 所以要切 K。本文件重点是 TILE_K=128 ----
//   TILE_K=128：一个缓冲 2*128*128 = 32768 B，两个正好 65536 B，刚好用满。
//   这一档是「双缓冲」最标准的落点，也是账算得最清楚的一档：
//
//     TILE_K   一个缓冲   两个缓冲   LDS 这一项允许的 block/CU（1 缓冲 / 2 缓冲）
//       256     64 KB     128 KB         1  /  放不下
//      *128     32 KB      64 KB         2  /  1        <- 本文件重点
//        64     16 KB      32 KB         4  /  2        <- 再切一刀的方向
//
//   双缓冲在 TK=128 上要做一笔明确的交换：
//       赚：这轮算 buf[cur] 的同时往 buf[nxt] 搬，发射点能提到 LDS 读之前，
//           整个窗口的 LDS 读 + 128 条 MMA 都用来盖搬运延迟；每窗还少一个 barrier。
//       亏：LDS 从 32KB 变 64KB，每 CU 驻留 block 从 2 个掉回 1 个 ——
//           也就是用「块内重叠」换掉了「跨 block 重叠」，而后者是硬件白送的。
//   所以同一个 TK 下必须有三个对照才分得开（表里 #0/#1/#2）：
//       顶部发射（无重叠，32KB）/ 晚发射（无重叠成本的重叠，32KB）/ 双缓冲（64KB）
//   「双缓冲 vs 顶部发射」给出重叠本身值多少，
//   「双缓冲 vs 晚发射」给出那 32KB 买得值不值。
//
//   ---- 代价说在前面：全局侧的连续段变短了 ----
//   一行在一个窗口里只取 TILE_K 个字节，所以段长就是 TILE_K：256 -> 128 -> 64。
//   本机实测过跨距读对段长极其敏感（README 第 5 节，跨距 8KB）：
//       段长  64 B -> 324 GB/s（线性峰值的 20%）
//       段长 512 B -> 1234 GB/s（77%）
//   切 K 赚驻留度和重叠、亏 DRAM 效率，**净账只能靠实测**。
//   256->128 那一刀是赚的（185 -> 267~322 TOPS，已测）；再往下切赚不赚，bench 说话。
//
//   ---- 顺带一件事：切到 TK=64 bank 冲突会自己消失 ----
//   冲突的根子是「行距是 128B 的整数倍，于是每行都从 bank 0 开始」。
//   TK=128 的行距正好一轮 —— 仍然每行都从 bank 0 开始，**所以 XOR 还得留着**
//   （表里 #3 就是 TK=128 不带 XOR 的对照，用来量冲突在双缓冲下还值多少）。
//   TK=64 的行距只有半轮：行 0 -> bank 0-3，行 1 -> bank 16-19，一奇一偶
//   劈成两个半轮，原始布局就已经是下限。这是 info 子命令那张表的内容。
//
//   ---- 八个组合 ----
//     #  TK   缓冲  发射方式      XOR   LDS      说明
//     0  128   1    顶部发射立刻等  有   32 KB   同 TK 的无重叠基准
//     1  128   1    晚发射          有   32 KB   不花 LDS 的重叠
//     2  128   2    双缓冲          有   64 KB  <- 本文件的重点
//     3  128   2    双缓冲          无   64 KB   量 bank 冲突在双缓冲下的代价
//     4  256   1    顶部发射立刻等  有   64 KB   上一篇的出发点
//     5   64   1    晚发射          有   16 KB   再切一刀：单缓冲
//     6   64   2    双缓冲          有   32 KB   再切一刀：驻留和重叠两样都要
//     7   64   3    三级流水        有   48 KB   更深的流水
//
//   编译：/opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_dbuf_256.cu -o mma_dbuf_256
//   运行：./mma_dbuf_256               八个组合各验一次正确性
//         ./mma_dbuf_256 info          LDS 预算 + bank 模型 + 指令条数（纯 host）
//         ./mma_dbuf_256 occ           每 CU 真实驻留 block 数（要 GPU）
//         ./mma_dbuf_256 b128 [M N K] [重复次数]
//                                      **只测 TK=128 的四个组合**，多轮取
//                                      最快/中位/最慢 + 离散度（默认 5 轮）
//         ./mma_dbuf_256 bench [M N K] 八个组合对照计时
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

#define TILE_M 128
#define TILE_N 128
#define THREADS 256
#define WAVE 64
#define WAVE_M 2
#define WAVE_N 2
#define WM (TILE_M / WAVE_M)  // 64
#define WN (TILE_N / WAVE_N)  // 64
#define MMA_M (WM / 16)       // 4
#define MMA_N (WN / 16)       // 4

#define BANKS 32                 // LDS bank 数
#define BANK_B 4                 // 每个 bank 的宽度
#define CHUNK 16                 // 一次 LDS / global 访问的宽度（b128）
#define TURN_B (BANKS * BANK_B)  // 128  一拍吐得出的字节数
#define LDS_CAP 65536            // 每 block 的 LDS 上限

#define MX_MIN(a, b) ((a) < (b) ? (a) : (b))
// 异或的周期：最多 8（一拍装得下的 16B 块数），且不能越过行边界。
//   TK=256/128 -> 7，TK=64 -> 3
#define SWZ_MASK_OF(TK) (MX_MIN(TURN_B / CHUNK, (TK) / CHUNK) - 1)

// 异步搬运（抄基线 kernel_test_v1.cu）
#define arrive_gvmcnt(count) __builtin_mxc_arrive(64 + (count))
// MACA_ICMP_SLT = 40，见 __clang_maca_device_library_decls.h
#define CMP_SLT 40

// 读侧寻址：逻辑 (行 r, 行内字节 k) -> LDS 字节下标
#define NOSWZ(TK, r, k) ((r) * (TK) + (k))
#define SWZ(TK, r, k)                                                          \
  ((r) * (TK) + (((((k) >> 4) ^ ((r) & SWZ_MASK_OF(TK)))) << 4) +              \
   ((k) & (CHUNK - 1)))
#define LDS_AT(SW, TK, r, k) ((SW) ? SWZ(TK, r, k) : NOSWZ(TK, r, k))
// 写侧：槽位 s 该去取逻辑第几块。异或是对合，所以和上面是同一个布局。
#define SRC_CHUNK(SW, TK, r, s) ((SW) ? ((s) ^ ((r) & SWZ_MASK_OF(TK))) : (s))

typedef __attribute__((ext_vector_type(4))) int i32x4;
typedef __attribute__((ext_vector_type(4))) int i8x16;

// -----------------------------------------------------------------------------
//  kernel
//    TK    k 窗口的宽度（字节）= 一个缓冲的 k 深度
//    NBUF  几个缓冲。1 = 单缓冲，2 = 双缓冲，>=3 = 多级流水
//    LATE  只在 NBUF==1 时有意义：发射挪到「LDS 读完 + barrier」之后
//    SW    LDS 布局：XOR / 原始
// -----------------------------------------------------------------------------
template <int TK, int NBUF, bool LATE, bool SW>
__global__ __launch_bounds__(THREADS) void mma_dbuf(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  constexpr int KCH = TK / 16 / 4;       // 每窗几个 k 组（c 的取值个数）
  constexpr int LRT = TK / CHUNK;        // 一行要几个线程来搬
  constexpr int RPR = THREADS / LRT;     // 一轮搬几行
  constexpr int RND = TILE_M / RPR;      // 几轮铺满 128 行
  constexpr int BUFB = 2 * TILE_M * TK;  // 一个缓冲（A 半 + B 半）多大
  constexpr int OPS = 2 * RND;           // 每线程每窗发几条 BSM
  // 等「最老那个窗口」落地时，允许 NBUF-2 个更晚的窗口还在飞。
  // NBUF<=2 时这就是 gvmcnt(0)，语义最干净；NBUF>=3 依赖 gvmcnt 按发射序回落，
  // 这一条**没有文档依据**，所以 NBUF=3 的组合由正确性检查来裁决。
  //
  // 注意排空阶段：循环尾部不再有新发射，在飞的窗口不足 NBUF-2 个，
  // 这时候 gvmcnt((NBUF-2)*OPS) 会立刻返回，于是**读到还没落地的数据**。
  // （host 上逐步模拟过，nwin=1..40 全错，不是只有最后一两轮。）
  // 所以流水满的时候用 WAITC，排空阶段降回 gvmcnt(0)。两个实参都是编译期常量。
  constexpr int WAITC = (NBUF >= 2) ? (NBUF - 2) * OPS : 0;

  static_assert(KCH >= 1, "TK 至少 64：一条 16B 读要覆盖 4 个 lane 组");
  static_assert(THREADS % LRT == 0, "线程数要能整除成整行");
  static_assert(TILE_M % RPR == 0, "128 行要能被一轮的行数整除");
  static_assert(BUFB * NBUF <= LDS_CAP, "LDS 放不下这么多缓冲");

  extern __shared__ int8_t smem[];

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

  // 搬运分工：LDS 落点 = i*(RPR*TK) + tid*16，按线程号连续 —— BSM 的硬性要求。
  // swizzle 因此只能做在「从 global 的哪一块取」上（异或对合，布局等价）。
  const int ld_row = tid / LRT;
  const int ld_slot = tid % LRT;

  // 取片段用的 lane 坐标
  const int mn = lane % 16;
  const int q = lane / 16;

  // 把第 win 个 k 窗口搬进第 buf 个缓冲
  auto load_win = [&](int buf, int win) {
    int8_t *As = smem + buf * BUFB;
    int8_t *Bs = As + TILE_M * TK;
#pragma unroll
    for (int i = 0; i < RND; ++i) {
      const int r = ld_row + i * RPR;
      // i*RPR 是 mask+1 的整数倍，所以这里和用 ld_row 算是一样的；
      // 写成 r 是为了换 TK / 换线程数时不用重新验这件事。
      const int j = SRC_CHUNK(SW, TK, r, ld_slot);
      const int dst = r * TK + ld_slot * CHUNK;
      const int koff = win * TK + j * CHUNK;
      __builtin_mxc_ldg_b128_bsm_predicator(
          As + dst, (void *)&A[(size_t)(block_row + r) * K + koff], 0, true,
          true, false, true, koff, K, CMP_SLT);
      __builtin_mxc_ldg_b128_bsm_predicator(
          Bs + dst, (void *)&B[(size_t)(block_col + r) * K + koff], 0, true,
          true, false, true, koff, K, CMP_SLT);
    }
  };

  const int nwin = K / TK;

  // ---- 预热 ----
  if (NBUF >= 2) {
    // 先把 NBUF-1 个窗口发出去，循环里每轮再补一个
#pragma unroll
    for (int s = 0; s < NBUF - 1; ++s)
      if (s < nwin) load_win(s, s);
  } else if (LATE) {
    load_win(0, 0);
  }

  int cur = 0;  // 本轮要算的缓冲号
  for (int tk = 0; tk < nwin; ++tk) {
    // 顶部发射的那一版：发完立刻等，中间什么都没有（这就是无重叠基准）
    if (NBUF == 1 && !LATE) load_win(0, tk);

    // ---- 等本窗落地 ----
    if (WAITC && tk + NBUF - 2 < nwin)
      arrive_gvmcnt(WAITC);  // 流水满：只等最老的那个，后面的继续飞
    else
      arrive_gvmcnt(0);      // NBUF<=2，或者流水在排空
    __builtin_mxc_barrier_inst();
    // barrier 过了，说明所有 wave 都读完了上一轮的缓冲 —— 那块地可以覆盖。

    // ---- 双缓冲 / 多级流水：现在就把下一个（下 NBUF-1 个）窗口发出去 ----
    //   发射点在「读 LDS」之前，所以它能和本窗的 LDS 读 + 全部 MMA 重叠，
    //   而且不需要「等所有人读完」那个 barrier —— 写的是另一块地。
    if (NBUF >= 2) {
      const int nx = tk + NBUF - 1;
      const int slot = (cur == 0) ? (NBUF - 1) : (cur - 1);
      if (nx < nwin) load_win(slot, nx);
    }

    const int8_t *As = smem + cur * BUFB;
    const int8_t *Bs = As + TILE_M * TK;

    // ---- LDS -> 寄存器 ----
    i8x16 af[MMA_M][KCH], bf[MMA_N][KCH];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCH; ++c) {
        const int k_off = q * CHUNK + c * 64;
        const int ra = wave_m * WM + s * 16 + mn;
        const int rb = wave_n * WN + s * 16 + mn;
        af[s][c] = *(const i8x16 *)&As[LDS_AT(SW, TK, ra, k_off)];
        bf[s][c] = *(const i8x16 *)&Bs[LDS_AT(SW, TK, rb, k_off)];
      }

    // ---- 单缓冲晚发射：LDS 这块地现在没人要了，可以直接往上覆盖 ----
    if (NBUF == 1 && LATE) {
      __builtin_mxc_barrier_inst();  // 确认所有 wave 都读完了
      if (tk + 1 < nwin) load_win(0, tk + 1);
    }

    // ---- MMA ----
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
      for (int n = 0; n < MMA_N; ++n)
#pragma unroll
        for (int c = 0; c < KCH; ++c)
#pragma unroll
          for (int w = 0; w < 4; ++w)
            acc[m][n] = __builtin_mxc_mma_16x16x16i8(af[m][c][w], bf[n][c][w],
                                                     acc[m][n]);

    // 顶部发射那一版要在这里拦一道，否则下一轮的搬运会盖掉还没读完的 LDS
    if (NBUF == 1 && !LATE) __builtin_mxc_barrier_inst();

    if (NBUF >= 2) cur = (cur + 1 == NBUF) ? 0 : cur + 1;
  }

  // ---- 写回 ----
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
//  组合登记表。前四个是 TK=128（本文件的重点），b128 子命令只跑这四个。
// =============================================================================
struct Var {
  const char *name;
  int tk, nbuf, late, sw;
};
static const Var kVar[8] = {
    {"TK=128 单缓冲顶部发射", 128, 1, 0, 1},
    {"TK=128 单缓冲晚发射", 128, 1, 1, 1},
    {"TK=128 双缓冲", 128, 2, 0, 1},
    {"TK=128 双缓冲(无XOR)", 128, 2, 0, 0},
    {"TK=256 单缓冲顶部发射", 256, 1, 0, 1},
    {"TK=64  单缓冲晚发射", 64, 1, 1, 1},
    {"TK=64  双缓冲", 64, 2, 0, 1},
    {"TK=64  三级流水", 64, 3, 0, 1},
};
#define NVAR 8
#define N128 4  // 前 N128 个是 TK=128
static int smem_of(int v) { return 2 * TILE_M * kVar[v].tk * kVar[v].nbuf; }

// =============================================================================
//  host 侧模型：不需要 GPU，也不需要 profiler。
//  bank 分布是地址的确定性函数，LDS 预算是算术。
// =============================================================================
static int mask_of(int tk) { return MX_MIN(TURN_B / CHUNK, tk / CHUNK) - 1; }

static int lds_at(int sw, int tk, int r, int k) {
  return sw ? (r * tk + ((((k >> 4) ^ (r & mask_of(tk)))) << 4) +
               (k & (CHUNK - 1)))
            : (r * tk + k);
}
static int src_chunk(int sw, int tk, int r, int s) {
  return sw ? (s ^ (r & mask_of(tk))) : s;
}

// 一次 16B 访问压 4 个连续 dword，落在 4 个连续 bank 上
static void add_access(int *hist, int byte_addr) {
  for (int d = 0; d < CHUNK / BANK_B; ++d)
    hist[(byte_addr / BANK_B + d) % BANKS]++;
}

// 一条 LDS 读指令：64 条 lane，(base, s, c) 固定
static void read_hist(int sw, int tk, int base, int s, int c, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int lane = 0; lane < WAVE; ++lane) {
    const int r = base + s * 16 + (lane % 16);
    const int k = (lane / 16) * CHUNK + c * 64;
    add_access(hist, lds_at(sw, tk, r, k));
  }
}

// 一条写指令：LDS 落点恒为按线程号连续，两种布局完全一样
static void write_hist(int tk, int i, int *hist) {
  const int lrt = tk / CHUNK, rpr = THREADS / lrt;
  memset(hist, 0, sizeof(int) * BANKS);
  for (int t = 0; t < WAVE; ++t)
    add_access(hist, (t / lrt + i * rpr) * tk + (t % lrt) * CHUNK);
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
static void print_hist(const char *tag, const int *h) {
  printf("  %s", tag);
  for (int b = 0; b < BANKS; ++b) printf("%3d", h[b]);
  printf("   用到 %2d/%d，最深 %2d -> %2d 拍\n", hist_used(h), BANKS,
         hist_max(h), hist_max(h));
}

// 布局自检：写进去的块 <-> 读出来的块必须对上，否则数值就是错的
static int layout_roundtrip(int sw, int tk) {
  const int cpr = tk / CHUNK;
  int bad = 0;
  for (int r = 0; r < TILE_M; ++r) {
    int holds[16], seen[16];
    for (int s = 0; s < cpr; ++s) holds[s] = src_chunk(sw, tk, r, s);
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < cpr; ++s) {
      if (holds[s] < 0 || holds[s] >= cpr) { ++bad; continue; }
      if (seen[holds[s]]++) ++bad;
    }
    for (int j = 0; j < cpr; ++j) {
      const int a = lds_at(sw, tk, r, j * CHUNK);
      if (a < r * tk || a >= (r + 1) * tk) { ++bad; continue; }
      if (holds[(a - r * tk) / CHUNK] != j) ++bad;
    }
  }
  return bad;
}

// 全局侧足迹自检：同一行的那几条 lane 取的还是那一段连续字节吗
static int global_footprint(int sw, int tk) {
  const int cpr = tk / CHUNK, rpr = THREADS / cpr;
  int bad = 0;
  for (int r = 0; r < rpr; ++r) {
    int seen[16];
    memset(seen, 0, sizeof(seen));
    for (int s = 0; s < cpr; ++s) {
      const int j = src_chunk(sw, tk, r, s);
      if (j < 0 || j >= cpr) { ++bad; continue; }
      if (seen[j]++) ++bad;
    }
    for (int j = 0; j < cpr; ++j)
      if (seen[j] != 1) ++bad;
  }
  return bad;
}

static void info_report(void) {
  const int floor_cycles = WAVE * CHUNK / TURN_B;  // 8
  int h[BANKS];

  printf("LDS：%d 个 bank × %d B = %d B/拍；每 block 上限 %d B\n", BANKS, BANK_B,
         TURN_B, LDS_CAP);
  printf("一条 wave 宽的 16B 访问要吐 %d × %d = %d B，下限 %d 拍\n", WAVE, CHUNK,
         WAVE * CHUNK, floor_cycles);

  printf("\n=== 重点：TILE_K=128 的 LDS 账 ===\n");
  printf("  一个缓冲 2 × %d × %d = %d B\n", TILE_M, 128, 2 * TILE_M * 128);
  printf("  两个缓冲 %d B = 正好顶满每 block 的 %d B 上限\n",
         2 * 2 * TILE_M * 128, LDS_CAP);
  printf("  -> 单缓冲时 LDS 允许 %d 个 block/CU，双缓冲时只剩 %d 个。\n",
         LDS_CAP / (2 * TILE_M * 128), LDS_CAP / (2 * 2 * TILE_M * 128));
  printf("  这就是双缓冲在这一档要付的全部代价：用跨 block 重叠换块内重叠。\n");

  printf("\n=== 切 K 的账：LDS 预算 ===\n");
  printf("  %-8s %10s %10s %10s %10s %12s\n", "TILE_K", "1 缓冲", "2 缓冲",
         "3 缓冲", "4 缓冲", "段长/行");
  for (int t = 256; t >= 64; t >>= 1) {
    printf("  %-8d", t);
    for (int nb = 1; nb <= 4; ++nb) {
      const int b = 2 * TILE_M * t * nb;
      if (b > LDS_CAP)
        printf("%10s", "放不下");
      else
        printf("%8dKB", b / 1024);
    }
    printf("%10d B\n", t);
  }
  printf("  LDS 是按 block 整块分配的，所以每一格都等价于「LDS 这一项允许\n");
  printf("  每 CU 驻留几个 block」= %d / 这一格。真实驻留还受寄存器约束，\n",
         LDS_CAP);
  printf("  用 occ 子命令查实测值。\n");

  printf("\n=== 切 K 的账：bank 冲突 ===\n");
  printf("  %-8s %8s %24s %24s\n", "TILE_K", "行距", "原始布局", "XOR 布局");
  for (int t = 256; t >= 64; t >>= 1) {
    const int kch = t / 16 / 4;
    int worst[2] = {0, 0}, used[2] = {0, 0};
    for (int sw = 0; sw < 2; ++sw)
      for (int base = 0; base < TILE_M; base += WM)
        for (int s = 0; s < MMA_M; ++s)
          for (int c = 0; c < kch; ++c) {
            read_hist(sw, t, base, s, c, h);
            if (hist_max(h) > worst[sw]) worst[sw] = hist_max(h);
            used[sw] = hist_used(h);
          }
    char c0[40], c1[40];
    snprintf(c0, sizeof(c0), "%2d 拍 %2d/%d bank %s", worst[0], used[0], BANKS,
             worst[0] == floor_cycles ? "下限" : "冲突");
    snprintf(c1, sizeof(c1), "%2d 拍 %2d/%d bank %s", worst[1], used[1], BANKS,
             worst[1] == floor_cycles ? "下限" : "冲突");
    printf("  %-8d %6.1f 轮 %24s %24s\n", t, (double)t / TURN_B, c0, c1);
  }
  printf("  行距是 128B 的整数倍 -> 每行都从 bank 0 开始 -> 冲突。\n");
  printf("  **TK=128 的行距正好一轮，所以 XOR 必须留着**（表里 #3 量它的代价）。\n");
  printf("  TK=64 的行距只有半轮，行号一奇一偶劈成两个半轮，原始布局就是下限。\n");

  printf("\n  TK=128 一条读指令 (base=0,s=0,c=0) 的 bank 深度：\n");
  printf("  bank");
  for (int b = 0; b < BANKS; ++b) printf("%3d", b);
  printf("\n");
  read_hist(0, 128, 0, 0, 0, h);
  print_hist("原始", h);
  read_hist(1, 128, 0, 0, 0, h);
  print_hist("XOR ", h);

  printf("\n=== 每个 k 窗口、每个 wave 的指令账 ===\n");
  printf("  %-8s %8s %8s %12s %12s\n", "TILE_K", "LDS 读", "MMA", "每线程 BSM",
         "窗口数(K=4096)");
  for (int t = 256; t >= 64; t >>= 1) {
    const int kch = t / 16 / 4;
    const int nread = (MMA_M + MMA_N) * kch, nmma = MMA_M * MMA_N * kch * 4;
    const int rnd = TILE_M / (THREADS / (t / CHUNK));
    printf("  %-8d %8d %8d %12d %12d\n", t, nread, nmma, 2 * rnd, 4096 / t);
  }
  printf("  切 K 不改变总工作量：窗口数翻倍、每窗的读和 MMA 各减半，\n");
  printf("  读:MMA 恒为 1:8。变的是 barrier 条数（跟窗口数成正比）和全局段长。\n");

  printf("\n=== 自检 ===\n");
  for (int t = 256; t >= 64; t >>= 1)
    for (int sw = 0; sw < 2; ++sw)
      printf("  TK=%-3d %-4s 布局往返 %s，全局足迹 %s\n", t, sw ? "XOR" : "原始",
             layout_roundtrip(sw, t) ? "对不上！" : "ok",
             global_footprint(sw, t) ? "变了！" : "ok");
  {
    int bad = 0;
    for (int t = 256; t >= 64; t >>= 1)
      for (int i = 0; i < TILE_M / (THREADS / (t / CHUNK)); ++i) {
        write_hist(t, i, h);
        if (hist_max(h) != floor_cycles) ++bad;
      }
    printf("  三种 TK 的写指令都在下限（%d 拍）：%s\n", floor_cycles,
           bad ? "不是！" : "ok");
  }
  {
    // BSM 的硬性前提：LDS 落点必须按线程号连续。
    // kernel 里写的是 r*TK + ld_slot*CHUNK，这里验它恒等于 tid*16 + i*RPR*TK。
    int bad = 0;
    for (int t = 256; t >= 64; t >>= 1) {
      const int lrt = t / CHUNK, rpr = THREADS / lrt;
      for (int i = 0; i < TILE_M / rpr; ++i)
        for (int tid = 0; tid < THREADS; ++tid) {
          const int r = tid / lrt + i * rpr;
          if (r * t + (tid % lrt) * CHUNK != tid * CHUNK + i * rpr * t) ++bad;
        }
    }
    printf("  LDS 落点 == tid*%d + i*(一轮的字节数)（BSM 要的连续性）：%s\n",
           CHUNK, bad ? "对不上！" : "ok");
  }

  printf("\n=== 八个组合 ===\n");
  printf("  %-24s %5s %5s %7s %5s %9s %12s\n", "组合", "TK", "缓冲", "发射",
         "XOR", "LDS", "LDS 允许");
  for (int v = 0; v < NVAR; ++v)
    printf("  %-24s %5d %5d %7s %5s %7dKB %9d 个%s\n", kVar[v].name, kVar[v].tk,
           kVar[v].nbuf,
           kVar[v].nbuf >= 2 ? "预取" : (kVar[v].late ? "晚" : "顶"),
           kVar[v].sw ? "有" : "无", smem_of(v) / 1024, LDS_CAP / smem_of(v),
           v < N128 ? "   <- b128 跑这四个" : "");
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

template <int TK, int NBUF, bool LATE, bool SW>
static int occupancy_of(void) {
  int blocks = -1;
  mcError_t e = mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, (const void *)mma_dbuf<TK, NBUF, LATE, SW>, THREADS,
      2 * TILE_M * TK * NBUF);
  return (e == mcSuccess) ? blocks : -1;
}

static void occ_all(int *out) {
  out[0] = occupancy_of<128, 1, false, true>();
  out[1] = occupancy_of<128, 1, true, true>();
  out[2] = occupancy_of<128, 2, false, true>();
  out[3] = occupancy_of<128, 2, false, false>();
  out[4] = occupancy_of<256, 1, false, true>();
  out[5] = occupancy_of<64, 1, true, true>();
  out[6] = occupancy_of<64, 2, false, true>();
  out[7] = occupancy_of<64, 3, false, true>();
}

static void occ_report(void) {
  int dev = 0;
  mcDeviceProp_t p;
  CHECK(mcGetDevice(&dev));
  CHECK(mcGetDeviceProperties(&p, dev));
  printf("设备 %s：%d 个 CU，每 block LDS 上限 %zu B\n\n", p.name,
         p.multiProcessorCount, (size_t)p.sharedMemPerBlock);
  int occ[NVAR];
  occ_all(occ);
  printf("  %-24s %10s %12s %12s\n", "组合", "LDS/block", "LDS 允许", "实测驻留");
  for (int v = 0; v < NVAR; ++v)
    printf("  %-24s %10d %10d 个 %10d 个\n", kVar[v].name, smem_of(v),
           LDS_CAP / smem_of(v), occ[v]);
  printf("\n实测值比「LDS 允许」小，说明卡在寄存器上而不是 LDS 上。\n");
  printf("TK=128 那两档（32KB vs 64KB）的差，就是双缓冲要付的代价。\n");
}

struct Run {
  double us;
  size_t bad;
};

template <int TK, int NBUF, bool LATE, bool SW>
static Run run_tmpl(const int8_t *dA, const int8_t *dB, int32_t *dC,
                    const int32_t *hR, int32_t *hC, int M, int N, int K,
                    int iters) {
  constexpr int SMEM = 2 * TILE_M * TK * NBUF;
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  Run r;
  r.us = 0;
  r.bad = 0;

  CHECK(mcMemset(dC, 0, (size_t)M * N * 4));
  mma_dbuf<TK, NBUF, LATE, SW><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
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
      mma_dbuf<TK, NBUF, LATE, SW><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_dbuf<TK, NBUF, LATE, SW><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
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

// 按登记表的下标分发，省得到处写模板实参
static Run run_idx(int v, const int8_t *dA, const int8_t *dB, int32_t *dC,
                   const int32_t *hR, int32_t *hC, int M, int N, int K,
                   int iters) {
  switch (v) {
    case 0: return run_tmpl<128, 1, false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 1: return run_tmpl<128, 1, true, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 2: return run_tmpl<128, 2, false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 3: return run_tmpl<128, 2, false, false>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 4: return run_tmpl<256, 1, false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 5: return run_tmpl<64, 1, true, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    case 6: return run_tmpl<64, 2, false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
    default: return run_tmpl<64, 3, false, true>(dA, dB, dC, hR, hC, M, N, K, iters);
  }
}

static int dcmp(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

// =============================================================================
//  b128：只测 TK=128 的四个组合，多轮。
//  为什么要多轮：上一篇里「TK=128 单缓冲晚发射」跨进程差过 20.6%，
//  而另外四个组合都 < 2.2%。单点数在这一档不可信，所以直接给分布。
// =============================================================================
static void bench128(const int8_t *dA, const int8_t *dB, int32_t *dC,
                     int32_t *hC, int M, int N, int K, int reps, int iters) {
  enum { MAXREP = 16 };
  if (reps > MAXREP) reps = MAXREP;
  double t[N128][MAXREP];

  for (int rep = 0; rep < reps; ++rep)
    for (int step = 0; step < N128; ++step) {
      // 奇数轮倒着跑，排掉「先跑的那个吃时钟爬坡」的顺序偏差
      const int v = (rep & 1) ? (N128 - 1 - step) : step;
      t[v][rep] = run_idx(v, dA, dB, dC, NULL, hC, M, N, K, iters).us;
    }

  const double flop = 2.0 * M * N * K;
  int occ[NVAR];
  occ_all(occ);

  printf("\n=== TK=128 重复测量：M=%d N=%d K=%d，%d 轮 × %d 次/轮 ===\n", M, N, K,
         reps, iters);
  printf("  每轮内部的顺序逐轮颠倒。单位 us。\n\n");
  printf("  %-24s", "组合");
  for (int rep = 0; rep < reps; ++rep) printf(" %8d", rep + 1);
  printf("\n");
  for (int v = 0; v < N128; ++v) {
    printf("  %-24s", kVar[v].name);
    for (int rep = 0; rep < reps; ++rep) printf(" %8.1f", t[v][rep]);
    printf("\n");
  }

  printf("\n  %-24s %9s %9s %9s %8s %9s %7s\n", "组合", "最快us", "中位us",
         "最慢us", "离散度", "TOPS(最快)", "驻留");
  double fast[N128];
  for (int v = 0; v < N128; ++v) {
    double s[MAXREP];
    memcpy(s, t[v], sizeof(double) * reps);
    qsort(s, reps, sizeof(double), dcmp);
    const double lo = s[0], mid = s[reps / 2], hi = s[reps - 1];
    fast[v] = lo;
    printf("  %-24s %9.1f %9.1f %9.1f %7.1f%% %9.1f %5d 个\n", kVar[v].name, lo,
           mid, hi, (hi - lo) / lo * 100.0, flop / (lo * 1e-6) / 1e12, occ[v]);
  }

  printf("\n=== TK=128 的三笔账（都用最快值）===\n");
  printf("  重叠本身值多少   双缓冲 vs 顶部发射   %.3fx\n", fast[0] / fast[2]);
  printf("  不花 LDS 的重叠  晚发射 vs 顶部发射   %.3fx\n", fast[0] / fast[1]);
  printf("  那 32KB 买得值吗 双缓冲 vs 晚发射     %.3fx  (驻留 %d -> %d)\n",
         fast[1] / fast[2], occ[1], occ[2]);
  printf("  bank 冲突的代价  双缓冲 有XOR vs 无   %.3fx\n", fast[3] / fast[2]);
  printf("\n  第三行 > 1 才说明「切 K 换双缓冲」这条路在 TK=128 上真的赢了；\n");
  printf("  <= 1 就说明 32KB 的 LDS 换不回它顶掉的那个 block/CU，\n");
  printf("  该往 TK=64 走（两份缓冲只占 32KB，驻留和重叠两样都要）——\n");
  printf("  那四个组合在 bench 子命令里。\n");
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "info") == 0) {
    info_report();
    return 0;
  }
  if (argc > 1 && strcmp(argv[1], "occ") == 0) {
    occ_report();
    return 0;
  }

  const int only128 = (argc > 1 && strcmp(argv[1], "b128") == 0);
  const int bench = only128 || (argc > 1 && strcmp(argv[1], "bench") == 0);
  int M = 256, N = 256, K = 256, reps = 5;
  if (bench && argc > 4) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
  }
  if (only128 && argc > 5) reps = atoi(argv[5]);
  if (M % TILE_M || N % TILE_N || K % 256) {
    fprintf(stderr, "M/N 需为 %d 的倍数，K 需为 256 的倍数\n", TILE_M);
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

  // 正确性：b128 只查那四个，其它情况查全部
  const int nchk = only128 ? N128 : NVAR;
  Run r[NVAR];
  for (int v = 0; v < nchk; ++v)
    r[v] = run_idx(v, dA, dB, dC, hR, hC, M, N, K, 0);

  printf("M=%d N=%d K=%d  grid %dx%d\n", M, N, K, N / TILE_N, M / TILE_M);
  if (verify) {
    printf("\n=== 正确性（对 CPU 参考）===\n");
    for (int v = 0; v < nchk; ++v)
      printf("  %-24s %s  (%zu/%zu 不符)\n", kVar[v].name,
             r[v].bad ? "FAIL" : "PASS", r[v].bad, szC);
    if (!only128) {
      printf("\n  只有三级流水 FAIL -> gvmcnt 不是按发射序回落的，那一档作废\n");
      printf("  （双缓冲用的是 gvmcnt(0)，不依赖这条假设）。\n");
    }
  } else {
    printf("正确性   尺寸过大，跳过 CPU 参考\n");
  }

  if (only128) {
    bench128(dA, dB, dC, hC, M, N, K, reps, 200);
  } else if (bench) {
    Run r1[NVAR], r2[NVAR];
    for (int v = 0; v < NVAR; ++v)
      r1[v] = run_idx(v, dA, dB, dC, NULL, hC, M, N, K, 200);
    for (int v = NVAR - 1; v >= 0; --v)
      r2[v] = run_idx(v, dA, dB, dC, NULL, hC, M, N, K, 200);

    const double flop = 2.0 * M * N * K;
    double best[NVAR];
    for (int v = 0; v < NVAR; ++v)
      best[v] = r1[v].us < r2[v].us ? r1[v].us : r2[v].us;
    int occ[NVAR];
    occ_all(occ);

    printf("\n=== 计时（两轮，第二轮顺序颠倒；下表取各自最快）===\n");
    printf("  %-24s %9s %9s %9s %9s %8s %7s\n", "组合", "第1轮us", "第2轮us",
           "取快us", "TOPS", "LDS/KB", "驻留");
    for (int v = 0; v < NVAR; ++v)
      printf("  %-24s %9.1f %9.1f %9.1f %9.1f %8d %5d 个\n", kVar[v].name,
             r1[v].us, r2[v].us, best[v], flop / (best[v] * 1e-6) / 1e12,
             smem_of(v) / 1024, occ[v]);

    printf("\n=== TK=128 内部（本文件重点）===\n");
    printf("  重叠本身值多少   双缓冲 vs 顶部发射   %.3fx\n", best[0] / best[2]);
    printf("  不花 LDS 的重叠  晚发射 vs 顶部发射   %.3fx\n", best[0] / best[1]);
    printf("  那 32KB 买得值吗 双缓冲 vs 晚发射     %.3fx  (驻留 %d -> %d)\n",
           best[1] / best[2], occ[1], occ[2]);
    printf("  bank 冲突的代价  双缓冲 有XOR vs 无   %.3fx\n", best[3] / best[2]);

    printf("\n=== 跨 TK ===\n");
    printf("  256 -> 128  同为顶部发射           %.3fx\n", best[4] / best[0]);
    printf("  128 -> 64   同为晚发射             %.3fx\n", best[1] / best[5]);
    printf("  64  双缓冲 vs 64 晚发射            %.3fx  (驻留 %d -> %d)\n",
           best[5] / best[6], occ[5], occ[6]);
    printf("  64  双缓冲 vs 128 双缓冲           %.3fx  (驻留 %d -> %d)\n",
           best[2] / best[6], occ[2], occ[6]);
    printf("  64  三级流水 vs 64 双缓冲          %.3fx  (驻留 %d -> %d)\n",
           best[6] / best[7], occ[6], occ[7]);
  }

  free(hA);
  free(hB);
  free(hC);
  if (hR) free(hR);
  CHECK(mcFree(dA));
  CHECK(mcFree(dB));
  CHECK(mcFree(dC));
  for (int v = 0; v < nchk; ++v)
    if (r[v].bad) return 1;
  return 0;
}
