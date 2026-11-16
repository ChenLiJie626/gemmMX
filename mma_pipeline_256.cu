// =============================================================================
// mma_pipeline_256.cu  ——  让搬运和计算重叠：单缓冲晚发射 vs 双缓冲
//
//   接着 mma_swizzle_256.cu 往下做。那边的结论是：XOR 布局把 LDS 读打到下限，
//   4096^3 拿到 1.85x；但**异步搬运单独上只值 1.1%**，因为代码长这样：
//
//       发异步读 -> arrive_gvmcnt(0) -> barrier -> 读LDS -> 256 条 MMA
//                   ^^^^^^^^^^^^^^^^ 发出去和等回来之间什么都没有
//
//   要让异步搬运真的赚到钱，得在「发」和「等」之间塞进计算。两条路：
//
//     MODE 1  单缓冲晚发射：把发射挪到「LDS->寄存器读完 + 一个 barrier」之后。
//             读完之后 LDS 这块地就没人要了，可以直接往上覆盖，
//             MMA 全在寄存器上跑，和搬运重叠。**不要额外 LDS。**
//             代价：必须插那个 barrier，确认所有人都读完了才能覆盖。
//
//     MODE 2  双缓冲：这轮算 buf[cur]，同时往 buf[nxt] 搬。
//             不需要「等所有人读完」那个 barrier（写的是另一块地），
//             于是发射能提得更早，而且每窗少一个 barrier。
//             代价：LDS 要两份。
//
//   ---- 双缓冲的 LDS 账 ----
//   TILE_K = 256 时一个缓冲就是 2*128*256 = 65536 B，正好顶满 64KB 上限，
//   **双缓冲放不下**。所以 MODE 2 必须把 TILE_K 砍到 128：
//       一个缓冲 2*128*128 = 32768 B，两个 65536 B，正好用满。
//   砍 TILE_K 不改变总工作量（窗口数翻倍、每窗 MMA 减半），
//   但 barrier 数量翻倍 —— 所以必须单独拿 TK=128 的单缓冲做对照，
//   才分得清「缩 tile」和「双缓冲」各自的账。
//
//   三个 MODE × 两个 TILE_K，同一份模板实例化，除这两处外逐字相同。
//   布局永远是 XOR + BSM 异步搬运（那两件事在上一篇已经定下来了）。
//
//   编译：/opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_pipeline_256.cu -o mma_pipeline_256
//   运行：./mma_pipeline_256            五个组合各验一次正确性
//         ./mma_pipeline_256 info       LDS 预算 / 指令条数表（纯 host）
//         ./mma_pipeline_256 bench      五个组合对照计时（可加 M N K）
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
#define CHUNK 16              // 一次 LDS / global 访问的宽度
#define LDS_CAP 65536         // 每 block 的 LDS 上限

// 异步搬运（抄基线 kernel_test_v1.cu）
#define arrive_gvmcnt(count) __builtin_mxc_arrive(64 + (count))
// MACA_ICMP_SLT = 40，见 __clang_maca_device_library_decls.h
#define CMP_SLT 40

// 搬运模式
#define MODE_NOOVERLAP 0  // 发射在循环顶，立刻等 —— 没有重叠（上一篇的样子）
#define MODE_LATE 1       // 单缓冲晚发射 —— 读完 + barrier 之后再发
#define MODE_DBUF 2       // 双缓冲 —— 这轮算 cur，同时往 nxt 搬

typedef __attribute__((ext_vector_type(4))) int i32x4;
typedef __attribute__((ext_vector_type(4))) int i8x16;

// 读侧寻址：逻辑 (行 r, 行内字节 k) -> LDS 字节下标（XOR 布局）
// 写侧落点必须按线程号连续（BSM 要求），所以异或做在全局侧的块号上。
#define SWZ(TK, r, k)                                                          \
  ((r) * (TK) + (((((k) >> 4) ^ ((r) & 7))) << 4) + ((k) & (CHUNK - 1)))

// -----------------------------------------------------------------------------
template <int TK, int MODE>
__global__ __launch_bounds__(THREADS) void mma_pipe(
    const int8_t *__restrict__ A, const int8_t *__restrict__ B,
    int32_t *__restrict__ C, int M, int N, int K) {
  constexpr int KCH = TK / 16 / 4;        // 每窗几个 k 组
  constexpr int LRT = TK / CHUNK;         // 一行要几个线程来搬
  constexpr int RPR = THREADS / LRT;      // 一轮搬几行
  constexpr int RND = TILE_M / RPR;       // 几轮铺满 128 行
  constexpr int BUFB = 2 * TILE_M * TK;   // 一个缓冲（A 半 + B 半）多大
  constexpr int NBUF = (MODE == MODE_DBUF) ? 2 : 1;

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

  // 搬运分工：LDS 落点 = i*(RPR*TK) + tid*16，按线程号连续（BSM 要的）
  const int ld_row = tid / LRT;
  const int ld_slot = tid % LRT;
  const int ld_chunk = ld_slot ^ (ld_row & 7);  // 全局侧取第几块（XOR 在这）

  // 取片段用的 lane 坐标
  const int mn = lane % 16;
  const int q = lane / 16;

  // 把一个 k 窗口搬进 buf
  auto load_win = [&](int buf, int win) {
    int8_t *As = smem + buf * BUFB;
    int8_t *Bs = As + TILE_M * TK;
#pragma unroll
    for (int i = 0; i < RND; ++i) {
      const int r = ld_row + i * RPR;
      const int dst = r * TK + ld_slot * CHUNK;
      const int koff = win * TK + ld_chunk * CHUNK;
      __builtin_mxc_ldg_b128_bsm_predicator(
          As + dst, (void *)&A[(size_t)(block_row + r) * K + koff], 0, true,
          true, false, true, koff, K, CMP_SLT);
      __builtin_mxc_ldg_b128_bsm_predicator(
          Bs + dst, (void *)&B[(size_t)(block_col + r) * K + koff], 0, true,
          true, false, true, koff, K, CMP_SLT);
    }
  };

  const int nwin = K / TK;

  // MODE 1/2 先把第 0 窗预取出去，循环里就只剩「等上一次发的」
  if (MODE != MODE_NOOVERLAP) load_win(0, 0);

  for (int tk = 0; tk < nwin; ++tk) {
    const int cur = (MODE == MODE_DBUF) ? (tk & 1) : 0;

    if (MODE == MODE_NOOVERLAP) load_win(0, tk);

    arrive_gvmcnt(0);              // 等这一窗到齐
    __builtin_mxc_barrier_inst();  // barrier A：搬运对所有人可见

    // 双缓冲的本钱就在这：写的是另一块地，**读之前就能发**。
    // 上一轮读的那块地（就是 nxt）已经被 barrier A 挡在身后了，不会撞。
    if (MODE == MODE_DBUF)
      if (tk + 1 < nwin) load_win((tk + 1) & 1, tk + 1);

    // ---- LDS -> 寄存器 ----
    const int8_t *As = smem + cur * BUFB;
    const int8_t *Bs = As + TILE_M * TK;
    i8x16 af[MMA_M][KCH], bf[MMA_N][KCH];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCH; ++c) {
        const int k_off = q * 16 + c * 64;
        const int ra = wave_m * WM + s * 16 + mn;
        const int rb = wave_n * WN + s * 16 + mn;
        af[s][c] = *(const i8x16 *)&As[SWZ(TK, ra, k_off)];
        bf[s][c] = *(const i8x16 *)&Bs[SWZ(TK, rb, k_off)];
      }

    // ---- 单缓冲只能晚发：要覆盖同一块地，必须先确认所有人都读完了 ----
    if (MODE == MODE_LATE) {
      __builtin_mxc_barrier_inst();  // barrier B
      if (tk + 1 < nwin) load_win(0, tk + 1);
    }

    // ---- MMA：纯寄存器，和上面刚发出去的搬运重叠 ----
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

    // 没有重叠的那版，读和算都做完了才能覆盖
    if (MODE == MODE_NOOVERLAP) __builtin_mxc_barrier_inst();
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
  (void)NBUF;
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

struct Variant {
  const char *name;
  int tk;
  int mode;
};
static const Variant kVar[5] = {
    {"TK=256 顶部发射(无重叠)", 256, MODE_NOOVERLAP},
    {"TK=256 单缓冲晚发射    ", 256, MODE_LATE},
    {"TK=128 顶部发射(无重叠)", 128, MODE_NOOVERLAP},
    {"TK=128 单缓冲晚发射    ", 128, MODE_LATE},
    {"TK=128 双缓冲          ", 128, MODE_DBUF},
};

static void info_report(void) {
  printf("每 block LDS 上限 %d B；TILE_M = TILE_N = %d，%d 线程\n\n", LDS_CAP,
         TILE_M, THREADS);
  printf("  %-24s %8s %8s %6s %8s %8s %8s\n", "组合", "缓冲数", "LDS用量",
         "窗口数", "MMA/窗", "读指令/窗", "barrier/窗");
  for (int v = 0; v < 5; ++v) {
    const int TK = kVar[v].tk, MODE = kVar[v].mode;
    const int bufb = 2 * TILE_M * TK;
    const int nbuf = (MODE == MODE_DBUF) ? 2 : 1;
    const int kch = TK / 16 / 4;
    const int nbar = (MODE == MODE_DBUF) ? 1 : 2;
    printf("  %-24s %8d %8d %6s %8d %8d %8d\n", kVar[v].name, nbuf, bufb * nbuf,
           "K/TK", MMA_M * MMA_N * kch * 4, 2 * MMA_M * kch, nbar);
  }
  printf("\n注：窗口数 = K/TILE_K，所以 TK 减半窗口数翻倍 ——\n");
  printf("    每窗的 MMA 和读指令都减半，总量不变，但 barrier 总数翻倍。\n");
  printf("\nTILE_K=256 一个缓冲就是 %d B，正好顶满 %d B 上限 -> 双缓冲放不下。\n",
         2 * TILE_M * 256, LDS_CAP);
  printf("TILE_K=128 一个缓冲 %d B，两个 %d B -> 正好用满。\n",
         2 * TILE_M * 128, 2 * 2 * TILE_M * 128);
}

// 每 CU 能同时驻留几个 block —— LDS 用量直接决定这个数
template <int TK, int MODE>
static int occupancy_of(void) {
  constexpr int SMEM = 2 * TILE_M * TK * ((MODE == MODE_DBUF) ? 2 : 1);
  int blocks = -1;
  mcError_t e = mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, (const void *)mma_pipe<TK, MODE>, THREADS, SMEM);
  return (e == mcSuccess) ? blocks : -1;
}

static void occ_report(void) {
  int dev = 0;
  mcDeviceProp_t p;
  CHECK(mcGetDevice(&dev));
  CHECK(mcGetDeviceProperties(&p, dev));
  printf("设备 %s：%d 个 CU，每 block LDS 上限 %zu B\n\n", p.name,
         p.multiProcessorCount, (size_t)p.sharedMemPerBlock);
  const int occ[5] = {
      occupancy_of<256, MODE_NOOVERLAP>(), occupancy_of<256, MODE_LATE>(),
      occupancy_of<128, MODE_NOOVERLAP>(), occupancy_of<128, MODE_LATE>(),
      occupancy_of<128, MODE_DBUF>()};
  printf("  %-24s %10s %14s\n", "组合", "LDS/block", "驻留 block/CU");
  for (int v = 0; v < 5; ++v) {
    const int nbuf = (kVar[v].mode == MODE_DBUF) ? 2 : 1;
    printf("  %-24s %10d %14d\n", kVar[v].name, 2 * TILE_M * kVar[v].tk * nbuf,
           occ[v]);
  }
  printf("\nLDS 是按 block 整块分配的，所以 LDS 用量直接决定每 CU 驻留几个 block。\n");
}

struct Run {
  double us;
  size_t bad;
};

template <int TK, int MODE>
static Run run_one(const int8_t *dA, const int8_t *dB, int32_t *dC,
                   const int32_t *hR, int32_t *hC, int M, int N, int K,
                   int iters) {
  constexpr int BUFB = 2 * TILE_M * TK;
  constexpr int SMEM = BUFB * ((MODE == MODE_DBUF) ? 2 : 1);
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  Run r;
  r.us = 0;
  r.bad = 0;

  CHECK(mcMemset(dC, 0, (size_t)M * N * 4));
  mma_pipe<TK, MODE><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) {
    fprintf(stderr, "launch error (TK=%d MODE=%d): %s\n", TK, MODE,
            mcGetErrorString(le));
    exit(1);
  }
  CHECK(mcMemcpy(hC, dC, (size_t)M * N * 4, mcMemcpyDeviceToHost));

  if (hR)
    for (size_t i = 0; i < (size_t)M * N; ++i)
      if (hC[i] != hR[i]) ++r.bad;

  if (iters) {
    for (int i = 0; i < 20; ++i)
      mma_pipe<TK, MODE><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
    CHECK(mcDeviceSynchronize());
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    CHECK(mcEventRecord(e0, 0));
    for (int i = 0; i < iters; ++i)
      mma_pipe<TK, MODE><<<grid, blk, SMEM>>>(dA, dB, dC, M, N, K);
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

// 把 5 个实例按 kVar 的顺序跑一遍
static void run_all(const int8_t *dA, const int8_t *dB, int32_t *dC,
                    const int32_t *hR, int32_t *hC, int M, int N, int K,
                    int iters, Run *out, int reverse) {
  for (int s = 0; s < 5; ++s) {
    const int v = reverse ? 4 - s : s;
    switch (v) {
      case 0: out[0] = run_one<256, MODE_NOOVERLAP>(dA, dB, dC, hR, hC, M, N, K, iters); break;
      case 1: out[1] = run_one<256, MODE_LATE>(dA, dB, dC, hR, hC, M, N, K, iters); break;
      case 2: out[2] = run_one<128, MODE_NOOVERLAP>(dA, dB, dC, hR, hC, M, N, K, iters); break;
      case 3: out[3] = run_one<128, MODE_LATE>(dA, dB, dC, hR, hC, M, N, K, iters); break;
      case 4: out[4] = run_one<128, MODE_DBUF>(dA, dB, dC, hR, hC, M, N, K, iters); break;
    }
  }
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

  const int bench = (argc > 1 && strcmp(argv[1], "bench") == 0);
  int M = 256, N = 256, K = 256;
  if (bench && argc > 4) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
  }
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

  const int iters = bench ? 200 : 0;
  Run r[5], r2[5];
  run_all(dA, dB, dC, hR, hC, M, N, K, iters, r, 0);

  printf("M=%d N=%d K=%d  grid %dx%d\n", M, N, K, N / TILE_N, M / TILE_M);
  if (verify) {
    printf("\n=== 正确性（对 CPU 参考）===\n");
    for (int i = 0; i < 5; ++i)
      printf("  %-24s %s  (%zu/%zu 不符)\n", kVar[i].name,
             r[i].bad ? "FAIL" : "PASS", r[i].bad, szC);
  } else {
    printf("正确性   尺寸过大，跳过 CPU 参考\n");
  }

  if (bench) {
    run_all(dA, dB, dC, NULL, hC, M, N, K, iters, r2, 1);  // 顺序颠倒再来一轮
    const double flop = 2.0 * M * N * K;
    double best[5];
    for (int i = 0; i < 5; ++i)
      best[i] = r[i].us < r2[i].us ? r[i].us : r2[i].us;

    printf("\n=== 计时（两轮，第二轮顺序颠倒；下表取各自最快）===\n");
    printf("  %-24s %10s %10s %10s %8s %8s\n", "组合", "第1轮us", "第2轮us",
           "取快us", "TOPS", "相对");
    for (int i = 0; i < 5; ++i)
      printf("  %-24s %10.3f %10.3f %10.3f %8.1f %7.3fx\n", kVar[i].name,
             r[i].us, r2[i].us, best[i], flop / (best[i] * 1e-6) / 1e12,
             best[0] / best[i]);

    printf("\n=== 拆开看 ===\n");
    printf("  TK=256 下，晚发射值多少            %.3fx\n", best[0] / best[1]);
    printf("  TK=128 下，晚发射值多少            %.3fx\n", best[2] / best[3]);
    printf("  只把 TK 从 256 砍到 128（都不重叠） %.3fx\n", best[0] / best[2]);
    printf("  TK=128 下，双缓冲比单缓冲晚发射     %.3fx\n", best[3] / best[4]);
    printf("  最好的那个 vs 起点                  %.3fx\n",
           best[0] / (best[4] < best[1] ? best[4] : best[1]));
  }

  free(hA);
  free(hB);
  free(hC);
  if (hR) free(hR);
  CHECK(mcFree(dA));
  CHECK(mcFree(dB));
  CHECK(mcFree(dC));
  for (int i = 0; i < 5; ++i)
    if (r[i].bad) return 1;
  return 0;
}
