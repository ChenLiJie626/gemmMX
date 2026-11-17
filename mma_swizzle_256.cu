// =============================================================================
// mma_swizzle_256.cu  ——  C600U / xcore1600，INT8 K32 MMA + XOR swizzle
//
//   C500 的 K16 MMA 每 lane 输入 1 个 int32；C600U 的 K32 MMA 输入 2 个。
//   保留 wave64 和 16B LDS 读，把相邻 int32 配成 {v[2*w], v[2*w+1]}。
//   K 置换为 k = 64*c + 16*q + 8*w + j (j=0..7)，A/B 使用同一置换。
//   512 线程 = 8 wave，按 2×4 划分 256×256 输出；每 wave 负责 128×64。
//   每个 K=256 窗口每 wave 为 8*4*4*2 = 256 条 MMA。
//   A/B 各占 64KiB，合计 128KiB LDS；先读入所有 c 片段，再集中执行 MMA。
//
//   模板的两个开关：
//     SW    = false / true    LDS 布局：原始 / XOR
//     ASYNC = false / true    搬运：普通指针拷贝 / 基线的 BSM 异步搬运
//   当前可执行程序仅实例化并测试 <true,true>：异步搬运 + XOR 布局。
//
//   ---- 冲突在哪 ----
//   以下沿用 32 bank × 4B 的参考地址模型，不代表 C600U 实测周期。
//   假设每 bank 每轮服务一个 dword，一轮共 32 * 4 = 128 字节。
//   一行 TILE_K = 256 字节，正好是 128 的整数倍 —— 每一行都从 bank 0 开始。
//   取片段时 16 条 lane 读 16 个不同的行、行内偏移却相同（同一个 q），
//   于是 64 条 lane 只用到 16 个 bank，每 bank 深度 16 dword，模型下限为 8。
//
//   ---- XOR 怎么修 ----
//   一行 256B = 16 个 16B 块。让逻辑第 j 块住到第 (j ^ (r & 7)) 个槽位，
//   相邻 8 行被错开到 8 个槽位，32 个 bank 全用上，模型深度降到 8。
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
//   编译：mxcc -x maca -O3 -std=c++17 --offload-arch=xcore1600 \
//           mma_swizzle_256.cu -o mma_swizzle_256
//   运行：./mma_swizzle_256            默认 4096³，全量验证 + 耗时/TOPS/MMA利用率
//         ./mma_swizzle_256 M N K     指定尺寸，直接计时并输出利用率
//         ./mma_swizzle_256 check M N K  指定尺寸全量正确性，不计时
//         ./mma_swizzle_256 trace     512³ 流水图入口：CPU 全量参考，仅一次 GEMM
//         ./mma_swizzle_256 info       设备属性 + 异步 XOR 的实际驻留数
//         ./mma_swizzle_256 banks      bank 直方图 + 布局自检（纯 host，不用 GPU）
//         ./mma_swizzle_256 bench      异步 XOR 计时（可加 M N K）
//         ./mma_swizzle_256 ceiling    只测同设备纯 K32 MMA 参照吞吐
//         ./mma_swizzle_256 bench M N K --peak-tops P  使用指定的参照 TOPS
//   MMA利用率 = [2*M*N*K/(time_us*1e6)] / reference_TOPS * 100%。
//   默认 reference_TOPS 由本程序实测纯 MMA 得到，非硬件 pipe-active 计数器。
// =============================================================================
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <cmath>

#if defined(__MACA_ARCH__) && (__MACA_ARCH__ < 1600 || __MACA_ARCH__ >= 1700)
#error "C600U kernel requires --offload-arch=xcore1600"
#endif

#define CHECK(x)                                                               \
  do {                                                                         \
    mcError_t e_ = (x);                                                        \
    if (e_ != mcSuccess) {                                                     \
      fprintf(stderr, "[%s:%d] %s -> %s\n", __FILE__, __LINE__, #x,            \
              mcGetErrorString(e_));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- C600U：256×256×256 tile，512 线程，2×4 个 wave ----
#define TILE_M 256
#define TILE_N 256
#define TILE_K 256
#define THREADS 512
#define WAVE 64
#define WAVE_M 2
#define WAVE_N 4
#define WM (TILE_M / WAVE_M)              // 128
#define WN (TILE_N / WAVE_N)              // 64
#define MMA_M (WM / 16)                   // 8
#define MMA_N (WN / 16)                   // 4
#define KCHUNK (TILE_K / 16 / 4)          // 4
#define LDG_ROW_THREADS 16
#define LDG_ROWS (THREADS / LDG_ROW_THREADS)    // 32 行 / 搬运轮
#define COPY_ROUNDS (TILE_M / LDG_ROWS)        // 8 轮 / K 窗口
#define SMEM_BYTES ((TILE_M + TILE_N) * TILE_K) // 128KiB
#define MMA_K 32
#define MMA_WORDS (MMA_K / (WAVE / 16) / 4)  // 2 int32 / lane / operand
#define MMA_PER_LOAD (4 / MMA_WORDS)         // 2 次 K32 / 16B LDS 读

static_assert(THREADS == WAVE * WAVE_M * WAVE_N, "wave grid must cover block");
static_assert(TILE_M % WAVE_M == 0 && TILE_N % WAVE_N == 0 &&
              WM % 16 == 0 && WN % 16 == 0, "wave tiles must contain whole MMA tiles");
static_assert(TILE_M == TILE_N && TILE_M % LDG_ROWS == 0 &&
              LDG_ROW_THREADS * 16 == TILE_K && LDG_ROWS % 8 == 0,
              "cooperative copy must cover both tiles and preserve the XOR mask");
static_assert(MMA_WORDS == 2 && KCHUNK * MMA_PER_LOAD * MMA_K == TILE_K,
              "K32 fragments must cover the complete K tile");

// ---- 参考 bank 地址模型；硬件服务粒度、延迟必须另行测量 ----
#define BANKS 32
#define BANK_B 4
#define CHUNK 16                          // 一次 LDS 访问的宽度
#define TURN_B (BANKS * BANK_B)           // 128  一拍吐得出的字节数
#define CHUNKS_PER_TURN (TURN_B / CHUNK)  // 8    一拍装得下几个 16B 块
#define SWZ_MASK (CHUNKS_PER_TURN - 1)    // 7

// ---- 异步搬运：使用具名计数器，避免沿用 C500 的 arrive 立即数编码 ----
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
typedef __attribute__((ext_vector_type(2))) int i32x2;  // K32 输入：8B / lane
typedef __attribute__((ext_vector_type(4))) int i8x16;  // 16B = 2 个 K32 MMA 片段

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
  const int ld_row = tid / LDG_ROW_THREADS;   // 0..31  本轮搬第几行
  const int ld_slot = tid % LDG_ROW_THREADS;  // 0..15  LDS 落点的槽位，不能动
  // 这个槽位该装逻辑第几块。i 只会让行号加 32 的倍数，不影响 r & 7。
  const int ld_chunk = SRC_CHUNK(SW, ld_row, ld_slot);

  // 取片段用的 lane 坐标
  const int mn = lane % 16;
  const int q = lane / 16;

  const int k_tiles = K / TILE_K;
  for (int tk = 0; tk < k_tiles; ++tk) {
    // 第 2 窗起，覆盖 LDS 前先等所有 wave 消费完上一窗口。
    // tk 对整个 block 一致；首窗不等，最后一窗之后也没有多余会合。
    if (tk != 0) {
      if (ASYNC)
        __builtin_mxc_barrier_inst();
      else
        __syncthreads();
    }
    // ---- 1. global -> LDS ----
    //   LDS 落点 r * TILE_K + ld_slot * 16 == i * 8192 + tid * 16，按线程号连续，
    //   这正是 BSM 的硬性要求；swizzle 体现在 ld_chunk（全局侧取第几块）。
#pragma unroll
    for (int i = 0; i < COPY_ROUNDS; ++i) {
      const int r = ld_row + i * LDG_ROWS;
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
      // 由 global->LDS 异步请求的完成等待保证搬运就绪，再让 block 会合。
      // 这里无需额外显式 arrive_bsmcnt；LDS 输入依赖由编译器处理。
      __builtin_mxc_arrive_gvmcnt(0);
      __builtin_mxc_barrier_inst();
    } else {
      __syncthreads();
    }

    // ---- 2. LDS -> 寄存器：先准备完整的 af[8][4]、bf[4][4] ----
    // 第一维选择 A/B 行小块，第二维选择 c，向量分量再选择四个 int32 word。
    i8x16 af[MMA_M][KCHUNK], bf[MMA_N][KCHUNK];
#pragma unroll
    for (int s = 0; s < MMA_M; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * 16 + c * 64;
        const int ra = wave_m * WM + s * 16 + mn;
        af[s][c] = *(const i8x16 *)&As[LDS_AT(SW, ra, k_off)];
      }
#pragma unroll
    for (int s = 0; s < MMA_N; ++s)
#pragma unroll
      for (int c = 0; c < KCHUNK; ++c) {
        const int k_off = q * 16 + c * 64;
        const int rb = wave_n * WN + s * 16 + mn;
        bf[s][c] = *(const i8x16 *)&Bs[LDS_AT(SW, rb, k_off)];
      }

    // ---- 3. MMA：集中遍历 (m,n,c,w)，共 256 条 / wave / K 窗口 ----
    // 源码读入和计算分开；具体 LDS/MMA 指令的穿插仍由编译器调度。
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
      for (int n = 0; n < MMA_N; ++n)
#pragma unroll
        for (int c = 0; c < KCHUNK; ++c)
#pragma unroll
          for (int w = 0; w < MMA_PER_LOAD; ++w) {
            const i32x2 a = {af[m][c][2 * w], af[m][c][2 * w + 1]};
            const i32x2 b = {bf[n][c][2 * w], bf[n][c][2 * w + 1]};
            acc[m][n] = __builtin_mxc_mma_16x16x32i8(a, b, acc[m][n]);
          }
    // MMA 后不额外等待：下一轮入口保护 LDS 复用，最后一轮直接写回私有 acc。
  }

  // ---- 4. 写回：每线程 8*4*4 = 128 个输出，512 线程覆盖 256*256 个元素 ----
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

template <bool SW, bool ASYNC>
static void configure_kernel(void) {
  CHECK(mcFuncSetAttribute((const void *)mma_256<SW, ASYNC>,
                          mcFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
}

// 独立标量参考：16×16 输出/block，K 按 32 分组；不用 MMA 或 XOR。
__global__ void scalar_gemm_reference(const int8_t *A, const int8_t *B,
                                      int32_t *R, int M, int N, int K) {
  __shared__ int32_t a[16][33], b[32][17];
  const int tid = threadIdx.x;
  const int row = blockIdx.y * 16 + tid / 16;
  const int col = blockIdx.x * 16 + tid % 16;
  int32_t sum = 0;
  for (int base = 0; base < K; base += 32) {
#pragma unroll
    for (int index = tid; index < 16 * 32; index += 256) {
      const int r = index / 32, k = index % 32;
      const int ar = blockIdx.y * 16 + r;
      const int br = blockIdx.x * 16 + r;
      a[r][k] = ar < M && base + k < K ? (int32_t)A[(size_t)ar*K+base+k] : 0;
      b[k][r] = br < N && base + k < K ? (int32_t)B[(size_t)br*K+base+k] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 32; ++k) sum += a[tid/16][k] * b[k][tid%16];
    __syncthreads();
  }
  if (row < M && col < N) R[(size_t)row*N+col] = sum;
}

// 纯 K32 MMA 参照：所有输入常驻寄存器，输出 checksum 防止计算被消除。
template <int ACC>
__global__ __launch_bounds__(THREADS) void mma_throughput_reference(
    int32_t *out, int iterations, int32_t av, int32_t bv) {
  const i32x2 a = {av, av}, b = {bv, bv};
  i32x4 acc[ACC];
#pragma unroll
  for (int j = 0; j < ACC; ++j) acc[j] = (i32x4){j,j,j,j};
#pragma unroll 1
  for (int i = 0; i < iterations; ++i)
#pragma unroll
    for (int j = 0; j < ACC; ++j)
      acc[j] = __builtin_mxc_mma_16x16x32i8(a,b,acc[j]);
  int32_t checksum = 0;
#pragma unroll
  for (int j = 0; j < ACC; ++j)
    checksum += acc[j][0]+acc[j][1]+acc[j][2]+acc[j][3];
  out[(size_t)blockIdx.x*blockDim.x+threadIdx.x] = checksum;
}

template <int ACC>
static double measure_mma_reference(void) {
  int device = 0, active = 0;
  mcDeviceProp_t prop;
  CHECK(mcGetDevice(&device));
  CHECK(mcGetDeviceProperties(&prop,device));
  CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &active,(const void *)mma_throughput_reference<ACC>,THREADS,0));
  if (active < 1) { fprintf(stderr,"纯 MMA 参照无法驻留\n"); exit(1); }
  const int blocks = prop.multiProcessorCount * active * 2;
  const int iterations = 2048, batch = 8;
  const size_t count = (size_t)blocks * THREADS;
  int32_t *out;
  CHECK(mcMalloc((void **)&out,count*sizeof(int32_t)));
  auto launch = [&]() {
    mma_throughput_reference<ACC><<<blocks,THREADS>>>(
        out,iterations,0x01010101,0x01010101);
  };
  launch();
  CHECK(mcGetLastError());
  CHECK(mcDeviceSynchronize());
  int32_t *host = (int32_t *)malloc(count*sizeof(int32_t));
  if (!host) { fprintf(stderr,"纯 MMA 校验缓冲分配失败\n"); exit(1); }
  CHECK(mcMemcpy(host,out,count*sizeof(int32_t),mcMemcpyDeviceToHost));
  const int32_t expected = ACC*4*32*iterations + 2*ACC*(ACC-1);
  for (size_t i = 0; i < count; ++i)
    if (host[i] != expected) {
      fprintf(stderr,"纯 MMA checksum 错误：index=%zu got=%d expected=%d\n",
              i,host[i],expected);
      exit(1);
    }
  free(host);
  for (int i = 0; i < 8; ++i) launch();
  CHECK(mcDeviceSynchronize());
  mcEvent_t begin,end;
  CHECK(mcEventCreate(&begin));
  CHECK(mcEventCreate(&end));
  double samples[5];
  for (int group = 0; group < 5; ++group) {
    CHECK(mcEventRecord(begin,0));
    for (int i = 0; i < batch; ++i) launch();
    CHECK(mcEventRecord(end,0));
    CHECK(mcEventSynchronize(end));
    CHECK(mcGetLastError());
    float ms = 0;
    CHECK(mcEventElapsedTime(&ms,begin,end));
    samples[group] = ms*1000.0/batch;
  }
  for (int i = 1; i < 5; ++i) {
    const double value = samples[i];
    int j = i;
    while (j > 0 && samples[j-1] > value) {
      samples[j] = samples[j-1]; --j;
    }
    samples[j] = value;
  }
  // 每条 wave 级 K32 MMA 为 2*16*16*32=16384 ops；不能再乘 64 lane。
  const double operations = 16384.0 * ACC * iterations * blocks * (THREADS/WAVE);
  if (samples[2] <= 0) { fprintf(stderr,"纯 MMA event 耗时无效\n"); exit(1); }
  const double tops = operations/(samples[2]*1e6);
  printf("MMA_REF acc=%d blocks=%d threads=%d iterations=%d median_us=%.6f "
         "TOPS=%.6f checksum=PASS\n",ACC,blocks,THREADS,iterations,samples[2],tops);
  CHECK(mcEventDestroy(begin));
  CHECK(mcEventDestroy(end));
  CHECK(mcFree(out));
  return tops;
}

static double calibrate_mma_reference(void) {
  const double candidates[] = {measure_mma_reference<4>(),
                                measure_mma_reference<16>(),
                                measure_mma_reference<32>()};
  double best = candidates[0];
  for (int i = 1; i < 3; ++i) if (candidates[i] > best) best = candidates[i];
  printf("纯 MMA 实测参照 = %.6f TOPS（取 3 组累加器配置的最高中位吞吐）\n",best);
  return best;
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

// 第 1 段的一条写指令：LDS 落点恒为 i*8192 + tid*16，两种布局完全一样。
static void write_hist(int i, int wave, int *hist) {
  memset(hist, 0, sizeof(int) * BANKS);
  for (int lane = 0; lane < WAVE; ++lane) {
    const int tid = wave * WAVE + lane;
    add_access(hist, (tid / LDG_ROW_THREADS + i * LDG_ROWS) * TILE_K +
                         (tid % LDG_ROW_THREADS) * CHUNK);
  }
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
  printf("   用到 %2d/%d bank，最大 dword 深度 %2d\n", hist_used(h), BANKS,
         hist_max(h));
}

// 布局自检：写进去的是什么，读出来的就该是什么
//   写：LDS (行 r, 槽位 s) 装的是逻辑第 src_chunk(r, s) 块
//   读：想要逻辑第 j 块时，去 lds_at(r, j*16) 那个槽位取
// 两者必须对上，否则数值就是错的。
static int layout_roundtrip(int sw, int rows) {
  int bad = 0;
  for (int r = 0; r < rows; ++r) {
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
  for (int ld_row = 0; ld_row < LDG_ROWS; ++ld_row) {
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

static int bank_report(void) {
  const int floor_cycles = WAVE * CHUNK / TURN_B;  // 64*16/128 = 8
  int h[BANKS];

  printf("参考地址模型，非 C600U 实测周期；未建模广播、指令拆分或端口。\n");
  printf("LDS：%d 个 bank × %d B = %d B/模型轮；一行 TILE_K = %d B = %d 轮\n", BANKS,
         BANK_B, TURN_B, TILE_K, TILE_K / TURN_B);
  printf("一条 wave 宽的 16B 访问要吐 %d × %d = %d B，均摊深度下限 %d\n\n", WAVE,
         CHUNK, WAVE * CHUNK, floor_cycles);

  printf("=== 第 2 段，一条 LDS 读指令 (base=0, s=0, c=0) 的 bank 深度 ===\n");
  printf("  %-6s", "bank");
  for (int b = 0; b < BANKS; ++b) printf("%3d", b);
  printf("\n");
  read_hist(0, 0, 0, 0, h);
  print_hist("原始", h);
  read_hist(1, 0, 0, 0, h);
  print_hist("XOR ", h);

  printf("\n=== 遍历 A/B 全部 wave 区域的 (base, s, c) 组合，取最坏 ===\n");
  for (int sw = 0; sw < 2; ++sw) {
    int worst = 0;
    for (int operand = 0; operand < 2; ++operand) {
      const int waves = operand ? WAVE_N : WAVE_M;
      const int span = operand ? WN : WM;
      const int fragments = operand ? MMA_N : MMA_M;
      for (int group = 0; group < waves; ++group)
        for (int s = 0; s < fragments; ++s)
          for (int c = 0; c < KCHUNK; ++c) {
            read_hist(sw, group * span, s, c, h);
            if (hist_max(h) > worst) worst = hist_max(h);
          }
    }
    printf("  %-4s 读：最大深度 %2d（模型下限 %d）-> %s\n", sw ? "XOR" : "原始",
           worst, floor_cycles, worst == floor_cycles ? "打到下限" : "有冲突");
  }
  {
    int worst = 0;
    for (int i = 0; i < COPY_ROUNDS; ++i)
      for (int wave = 0; wave < THREADS / WAVE; ++wave) {
        write_hist(i, wave, h);
        if (hist_max(h) > worst) worst = hist_max(h);
      }
    printf("  两种  写：最大深度 %2d（模型下限 %d）-> %s\n", worst, floor_cycles,
           worst == floor_cycles ? "打到下限" : "有冲突");
    printf("        （LDS 落点恒为 i*%d + tid*16，两种布局完全相同 ——\n",
           THREADS * CHUNK);
    printf("          这正是 BSM 异步搬运要求的「按线程号连续」）\n");
  }

  printf("\n=== 每个 k 窗口、每个 wave 的参考模型深度合计 ===\n");
  const int ninst = (MMA_M + MMA_N) * KCHUNK;  // A 32 条 + B 16 条
  printf("  原始 %d 条读 × 16 = %4d\n", ninst, ninst * 16);
  printf("  XOR  %d 条读 ×  8 = %4d   减少 %d（非硬件耗时）\n", ninst, ninst * 8,
         ninst * 8);

  printf("\n=== 布局自检 ===\n");
  int bad = 0;
  for (int sw = 0; sw < 2; ++sw) {
    const int errors = layout_roundtrip(sw, TILE_M) + layout_roundtrip(sw, TILE_N);
    bad += errors;
    printf("  %-4s A/B 写进去的块 <-> 读出来的块：%s（各 %d 行 × %d 块）\n",
           sw ? "XOR" : "原始", errors ? "对不上！" : "全部对上",
           TILE_M, TILE_K / CHUNK);
  }
  for (int sw = 0; sw < 2; ++sw)
    printf("  %-4s 全局侧 16 条 lane 的足迹：%s\n", sw ? "XOR" : "原始",
           global_footprint(sw) ? "变了！"
                                : "仍是那 256 个连续字节，只换了取数的人");
  return (bad || global_footprint(0) || global_footprint(1)) ? 1 : 0;
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
  configure_kernel<SW, ASYNC>();
  dim3 grid(N / TILE_N, M / TILE_M), blk(THREADS);
  Run r;
  r.us = 0;
  r.bad = 0;

  CHECK(mcMemset(dC, 0xa5, (size_t)M * N * 4));
  mma_256<SW, ASYNC><<<grid, blk, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
  CHECK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) {
    fprintf(stderr, "launch error: %s\n", mcGetErrorString(le));
    exit(1);
  }
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
  // 校验计时后的实际结果；传输与逐元素比较均不计入 kernel event 耗时。
  CHECK(mcMemcpy(hC, dC, (size_t)M * N * 4, mcMemcpyDeviceToHost));
  if (hR)
    for (size_t i = 0; i < (size_t)M * N; ++i)
      if (hC[i] != hR[i]) ++r.bad;
  return r;
}

static const char *kVariantName = "异步搬运 + XOR 布局";

static void occupancy_report(void) {
  configure_kernel<true, true>();
  int blocks = 0;
  CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, (const void *)mma_256<true, true>, THREADS, SMEM_BYTES));
  printf("  %-24s 驻留 %d block/AP，%d wave/AP\n", kVariantName, blocks,
         blocks * THREADS / WAVE);
}

static void device_report(void) {
  int device = 0;
  mcDeviceProp_t p;
  CHECK(mcGetDevice(&device));
  CHECK(mcGetDeviceProperties(&p, device));
  printf("设备 %d：%s，AP=%d，wave=%d，shared/block=%zu B\n", device, p.name,
         p.multiProcessorCount, p.warpSize, (size_t)p.sharedMemPerBlock);
  if (p.warpSize != WAVE) {
    fprintf(stderr, "此 kernel 需要 wave64，设备 wave=%d\n", p.warpSize);
    exit(1);
  }
  if (p.maxThreadsPerBlock < THREADS || (size_t)p.sharedMemPerBlock < SMEM_BYTES) {
    fprintf(stderr, "设备无法容纳 %d 线程、%d B shared memory 的 block\n",
            THREADS, SMEM_BYTES);
    exit(1);
  }
  printf("tile=%dx%dx%d，block=%d 线程=%d wave，wave grid=%dx%d，wave tile=%dx%d\n",
         TILE_M, TILE_N, TILE_K, THREADS, THREADS / WAVE, WAVE_M, WAVE_N, WM, WN);
  printf("C600U K32：%d MMA/wave/K256；LDS/block=%d B\n",
         MMA_M * MMA_N * KCHUNK * MMA_PER_LOAD, SMEM_BYTES);
  occupancy_report();
}

static int usage(const char *program) {
  fprintf(stderr,"用法：%s [bench|check] [M N K] [--peak-tops P]\n"
                 "      %s trace [512 512 512]\n"
                 "      %s [banks|info|ceiling]\n"
                 "默认：4096³，全量验证并输出耗时、TOPS、MMA利用率。\n",
          program,program,program);
  return 1;
}

static int positive_dimension(const char *text) {
  errno = 0;
  char *end = NULL;
  const long value = strtol(text,&end,10);
  if (errno || end == text || *end || value <= 0 || value > INT_MAX) {
    fprintf(stderr,"无效维度：%s\n",text); exit(1);
  }
  return (int)value;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "banks") == 0) {
    return bank_report();
  }
  if (argc == 2 && strcmp(argv[1], "info") == 0) {
    device_report();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "ceiling") == 0) {
    device_report();
    calibrate_mma_reference();
    return 0;
  }
  const bool check = argc > 1 && strcmp(argv[1],"check") == 0;
  const bool trace = argc > 1 && strcmp(argv[1],"trace") == 0;
  const bool bench = !check && !trace;
  int next = (argc > 1 && (check || trace || strcmp(argv[1],"bench") == 0)) ? 2 : 1;
  int M = trace ? 512 : (check ? 256 : 4096), N = M, K = M;
  if (next < argc && strncmp(argv[next],"--",2) != 0) {
    if (argc-next < 3) return usage(argv[0]);
    M = positive_dimension(argv[next]);
    N = positive_dimension(argv[next+1]);
    K = positive_dimension(argv[next+2]);
    next += 3;
  }
  double requested_tops = 0;
  if (next < argc) {
    if (!bench || argc-next != 2 || strcmp(argv[next],"--peak-tops") != 0)
      return usage(argv[0]);
    errno = 0;
    char *end = NULL;
    requested_tops = strtod(argv[next+1],&end);
    if (errno || end == argv[next+1] || *end || !std::isfinite(requested_tops) ||
        requested_tops <= 0) {
      fprintf(stderr,"参照 TOPS 必须为正的有限数\n"); return 1;
    }
    next += 2;
  }
  if (next != argc) return usage(argv[0]);
  if (trace && (M != 512 || N != 512 || K != 512)) {
    fprintf(stderr,"流水图入口仅允许 512 512 512，以限制采集数据量\n");
    return 1;
  }
  if (M <= 0 || N <= 0 || K <= 0 || M % TILE_M || N % TILE_N || K % TILE_K) {
    fprintf(stderr, "维度需为正数；M/N 需为 %d 的倍数，K 需为 %d 的倍数\n", TILE_M, TILE_K);
    return 1;
  }
  device_report();

  const size_t szA = (size_t)M * K, szB = (size_t)N * K, szC = (size_t)M * N;
  int8_t *hA = (int8_t *)malloc(szA), *hB = (int8_t *)malloc(szB);
  int32_t *hC = (int32_t *)malloc(szC * 4), *hR = (int32_t *)malloc(szC * 4);
  if (!hA || !hB || !hC || !hR) {
    fprintf(stderr,"host 缓冲分配失败\n"); return 1;
  }
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

  if (trace) {
    // 流水图只记录一次被测 GEMM；CPU 参考不产生额外 GPU 指令流。
    ref_gemm(hA,hB,hR,M,N,K);
    printf("TRACE_CONFIG M=512 N=512 K=512 variant=3 gemm_launches=1 CPU_reference=full\n");
  } else {
    // 普通正确性与计时模式使用独立标量 GPU 参考，不跳过大尺寸。
    CHECK(mcMemset(dC,0xa5,szC*sizeof(int32_t)));
    scalar_gemm_reference<<<dim3((N+15)/16,(M+15)/16),256>>>(dA,dB,dC,M,N,K);
    CHECK(mcGetLastError());
    CHECK(mcDeviceSynchronize());
    CHECK(mcMemcpy(hR,dC,szC*sizeof(int32_t),mcMemcpyDeviceToHost));
    if (M <= 1024 && N <= 1024 && K <= 1024) {
      ref_gemm(hA,hB,hC,M,N,K);
      for (size_t i = 0; i < szC; ++i)
        if (hC[i] != hR[i]) {
          fprintf(stderr,"GPU 标量参考与 CPU 不符：index=%zu\n",i); return 1;
        }
      printf("GPU 标量参考已通过全量 CPU 对照。\n");
    }
    printf("参考：独立标量 GPU，全量 %zu 个输出，检查位于计时区间外。\n",szC);
  }
  double reference_tops = 0;
  if (bench) {
    reference_tops = requested_tops > 0 ? requested_tops : calibrate_mma_reference();
    if (requested_tops > 0) printf("用户指定参照 = %.6f TOPS\n",reference_tops);
    printf("MMA利用率 = GEMM_TOPS / reference_TOPS * 100%%；"
           "这是时间法吞吐比，不是硬件 active 计数器。\n");
  }
  const int iters = bench ? 200 : 0;
  const Run r = run_one<true, true>(dA, dB, dC, hR, hC, M, N, K, iters);
  Run r2 = {};

  printf("M=%d N=%d K=%d  grid %dx%d  LDS %d B\n", M, N, K, N / TILE_N,
         M / TILE_M, SMEM_BYTES);
  printf("\n=== 正确性（第1轮结束后，全量 int32 比较）===\n");
  printf("  %-24s %s  (%zu/%zu 不符)\n",kVariantName,
         r.bad ? "FAIL" : "PASS",r.bad,szC);

  if (bench) {
    r2 = run_one<true, true>(dA, dB, dC, hR, hC, M, N, K, iters);

    const double flop = 2.0 * M * N * K;
    const double best = r.us < r2.us ? r.us : r2.us;

    printf("\n=== 异步搬运 + XOR 计时（两轮，取较快值）===\n");
    printf("参照吞吐 = %.6f TOPS，参照理想耗时 = %.6f us\n",
           reference_tops,flop/(reference_tops*1e6));
    printf("  %-24s %10s %10s %10s %10s %12s %8s\n", "组合", "第1轮us", "第2轮us",
           "取快us", "TOPS", "MMA利用率%", "全量校验");
    const double tops = flop/(best*1e6);
    const double utilization = tops/reference_tops*100.0;
    printf("  %-24s %10.3f %10.3f %10.3f %10.3f %12.3f %8s\n",kVariantName,
           r.us,r2.us,best,tops,utilization,
           (r.bad || r2.bad) ? "FAIL" : "PASS");
    printf("RESULT variant=3 M=%d N=%d K=%d time_us=%.6f TOPS=%.6f "
           "reference_TOPS=%.6f mma_utilization_pct=%.6f mismatches=%zu "
           "checked_outputs=%zu\n",M,N,K,best,tops,reference_tops,
           utilization,r.bad+r2.bad,szC*2);
  }

  free(hA);
  free(hB);
  free(hC);
  if (hR) free(hR);
  CHECK(mcFree(dA));
  CHECK(mcFree(dB));
  CHECK(mcFree(dC));
  if (r.bad || (bench && r2.bad)) return 1;
  return 0;
}
