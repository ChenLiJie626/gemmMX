// =============================================================================
// mma_teach.cu  --  沐曦 C500 int8 MMA 最小教学版
//
// 目标:用最少的代码把一条 MMA 指令的数据排布讲清楚。
//   * 1 个 wave(64 线程),输出 C 只有 16x16 —— 没有任何分块复杂度
//   * 每个线程一次搬 16B(16 个 int8),这是本教程的主线
//   * K = 256,正好 16 条 MMA 指令
//
// 约定与 kernel_test_v1.cu 一致:
//   A: M x K 行主序 int8
//   B: N x K 行主序 int8   <-- 注意 B 是 [n][k],host 侧已经转置好
//   C: M x N 行主序 int32
//   所以 A 和 B 形状完全同构,搬运和 LDS 布局用同一套代码,不需要转置。
//
// 编译: /opt/maca/mxgpu_llvm/bin/mxcc -x maca -O3 mma_teach.cu -o mma_teach
// 运行: ./mma_teach          正确性 + 把数据来源打成表
//       ./mma_teach 37       再详细拆解 C[2][5] 这一个元素(37 = 2*16+5)
// =============================================================================
#include <mc_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 16   // C 的行数 = 一个 MMA tile
#define N 16   // C 的列数
#define K 256  // 16 条 MMA 指令 x 每条 16 个 k

#define WAVE 64

// 累加器片段:4 个 int32
typedef __attribute__((ext_vector_type(4))) int i32x4;

// 一次 16B 访问:4 个 int32 = 4 个 MMA 片段
typedef __attribute__((ext_vector_type(4))) int i8x16;

// =============================================================================
// 片段布局 —— 全部摘自 SDK __clang_maca_mma_functions.h 的 load_matrix_sync
//
//  matrix_a / row_major    row = lane & 15        col = (lane >> 4) * 4
//  matrix_b / col_major    col = lane & 15        row = (lane >> 4) * 4
//  accumulator / int       col = lane & 15        row = (lane >> 4) * 4 + e
//
//  三者的 lane 分解完全一样:
//      lane & 15   -> A 的行 / B 的列 / C 的列
//      lane >> 4   -> k 组(A、B) / C 的行组
//
//  最容易弄错的一点:lane 供给的 A 行号 != 它持有的 C 行号。
//  lane L 供给 A[L&15][..],却接收 C[(L>>4)*4+e][L&15]。
//  也就是 L&15 在 A 里是行、在 C 里是列 —— 数据在 MMA 单元内部跨 lane 流动。
// =============================================================================

__global__ void mma_teach(const int8_t *__restrict__ A,
                          const int8_t *__restrict__ B,
                          int32_t *__restrict__ C) {
  // LDS:A 和 B 各 16 行 x 256 字节 = 4KB
  __shared__ int8_t As[M * K];
  __shared__ int8_t Bs[N * K];

  const int lane = threadIdx.x;         // 0..63,只有一个 wave

  // ---------------------------------------------------------------------------
  // 第 1 步:global -> LDS,每个线程一次搬 16B
  //
  //   一行是 K = 256 字节,16 个线程 x 16B 正好搬完一行。
  //   64 个线程 = 4 行 / 轮,16 行要 4 轮。
  //
  //   r = lane / 16  (0..3)  本轮搬第几行
  //   c = lane % 16         本轮搬该行的第几个 16B 块
  // ---------------------------------------------------------------------------
  const int r = lane / 16;
  const int c = lane % 16;

#pragma unroll
  for (int p = 0; p < 4; ++p) {          // 4 轮,每轮 4 行
    int row = p * 4 + r;
    int off = row * K + c * 16;          // 16B 对齐
    *(i8x16 *)&As[off] = *(const i8x16 *)&A[off];
    *(i8x16 *)&Bs[off] = *(const i8x16 *)&B[off];
  }
  __syncthreads();

  // ---------------------------------------------------------------------------
  // 第 2 步:LDS -> 寄存器,一次 16B 读 = 4 个 MMA 片段
  //
  //   lane 要读的行:A 用 lane&15(片段的 row),B 用 lane&15(片段的 col)。
  //   lane 要读的 k:(lane>>4)*16 + i*64,i = 0..3
  //
  //   一次 16B 读回来是 4 个 int32,每个 int32 就是一个 MMA 片段
  //   (4 个 int8 打包进 32 位寄存器 —— 这正是 builtin 的参数类型)。
  //   4 轮 i x 4 个 word = 16 个片段 = 16 条 MMA 指令,正好吃完 K=256。
  // ---------------------------------------------------------------------------
  const int mn = lane & 15;              // A 的行 / B 的列
  const int q = lane >> 4;               // k 组 0..3

  i8x16 af[4], bf[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int k_off = q * 16 + i * 64;
    af[i] = *(const i8x16 *)&As[mn * K + k_off];
    bf[i] = *(const i8x16 *)&Bs[mn * K + k_off];
  }

  // ---------------------------------------------------------------------------
  // 第 3 步:16 条 MMA
  //
  //   这里有一个关键的、也是最容易看漏的事实:
  //
  //   lane L 第 (i,w) 个片段持有的**逻辑 k** 是  (L>>4)*16 + i*64 + w*4,
  //   而硬件认为 lane L 在一条指令里提供的是该指令 16 宽窗口内的
  //   k0 = (L>>4)*4。两者**并不相等** —— 送进 MMA 的 k 顺序是被打乱的。
  //
  //   但这完全没关系,因为 k 方向是求和:只要 A 和 B 用**同一个置换**,
  //   每条指令算的都是同一组 (A,B) 配对,16 条指令的 k 集合合起来恰好
  //   覆盖 0..255 各一次(双射),总和就等于真正的点积。
  //
  //   这就是为什么这份代码既不需要转置 B、也不需要 swizzle:
  //   让 A 和 B 承受同样的错乱,错乱就互相抵消了。
  // ---------------------------------------------------------------------------
  i32x4 acc = (i32x4){0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int w = 0; w < 4; ++w)
      acc = __builtin_mxc_mma_16x16x16i8(af[i][w], bf[i][w], acc);

  // ---------------------------------------------------------------------------
  // 第 4 步:写回 C
  //   lane 持有 C 的连续 4 行、同一列:
  //     行 = (lane>>4)*4 + e      列 = lane&15
  //   同一行上 16 条 lane 的列号连续 -> 每行 16 个 int32 = 64B 连续,可合并。
  // ---------------------------------------------------------------------------
  const int cn = lane & 15;
  const int cm = (lane >> 4) * 4;
#pragma unroll
  for (int e = 0; e < 4; ++e) C[(cm + e) * N + cn] = acc[e];
}

// =============================================================================
// 来源探针:不算 GEMM,把每个 C 元素的生产者身份写出来
//   编码成一个 int: lane * 10 + e
// =============================================================================
__global__ void prov(int32_t *out) {
  const int lane = threadIdx.x;
  const int cn = lane & 15;
  const int cm = (lane >> 4) * 4;
#pragma unroll
  for (int e = 0; e < 4; ++e) out[(cm + e) * N + cn] = lane * 10 + e;
}

// =============================================================================
int main(int argc, char **argv) {
  int8_t *hA = (int8_t *)malloc(M * K), *hB = (int8_t *)malloc(N * K);
  int32_t *hC = (int32_t *)malloc(M * N * 4);
  int32_t *hP = (int32_t *)malloc(M * N * 4);

  // 用小整数,方便手算核对
  srand(7);
  for (int i = 0; i < M * K; ++i) hA[i] = (int8_t)((rand() % 7) - 3);
  for (int i = 0; i < N * K; ++i) hB[i] = (int8_t)((rand() % 7) - 3);

  int8_t *dA, *dB;
  int32_t *dC, *dP;
  mcMalloc((void **)&dA, M * K);
  mcMalloc((void **)&dB, N * K);
  mcMalloc((void **)&dC, M * N * 4);
  mcMalloc((void **)&dP, M * N * 4);
  mcMemcpy(dA, hA, M * K, mcMemcpyHostToDevice);
  mcMemcpy(dB, hB, N * K, mcMemcpyHostToDevice);

  mma_teach<<<1, WAVE>>>(dA, dB, dC);
  prov<<<1, WAVE>>>(dP);
  mcDeviceSynchronize();
  mcError_t err = mcGetLastError();
  if (err != mcSuccess) { printf("kernel 失败: %s\n", mcGetErrorString(err)); return 1; }
  mcMemcpy(hC, dC, M * N * 4, mcMemcpyDeviceToHost);
  mcMemcpy(hP, dP, M * N * 4, mcMemcpyDeviceToHost);

  // ---- 正确性:int32 无浮点误差,要求逐元素完全相等 ----
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t ref = 0;
      for (int k = 0; k < K; ++k) ref += (int32_t)hA[m * K + k] * (int32_t)hB[n * K + k];
      if (ref != hC[m * N + n]) ++bad;
    }
  printf("=== 正确性 ===\n");
  printf("C = A(16x256) x B^T(256x16),K=256 -> 16 条 MMA\n");
  printf("%s  (%d / %d 不符)\n\n", bad ? "FAIL" : "PASS", bad, M * N);

  // ---- 表 1:每个 C 元素由哪条 lane 算出 ----
  printf("=== 表 1:C[m][n] 由哪条 lane 持有 ===\n");
  printf("   n=   ");
  for (int n = 0; n < N; ++n) printf("%4d", n);
  printf("\n");
  for (int m = 0; m < M; ++m) {
    printf("m=%2d   ", m);
    for (int n = 0; n < N; ++n) printf("%4d", hP[m * N + n] / 10);
    printf("\n");
  }
  printf("\n看点:每 4 行一组用的是同一批 lane(行组 = lane>>4),\n");
  printf("      列号直接就是 lane&15。\n\n");

  // ---- 表 2:片段内的第几个分量 ----
  printf("=== 表 2:它是该 lane 累加器 i32x4 的第几个分量 e ===\n");
  printf("   n=   ");
  for (int n = 0; n < N; ++n) printf("%4d", n);
  printf("\n");
  for (int m = 0; m < M; ++m) {
    printf("m=%2d   ", m);
    for (int n = 0; n < N; ++n) printf("%4d", hP[m * N + n] % 10);
    printf("\n");
  }
  printf("\n看点:e 只随行走,不随列走 —— 一条 lane 的 4 个 int32 是\n");
  printf("      C 的连续 4 行、同一列。\n\n");

  // ---- 表 3:一条 lane 在 A / B / C 里分别碰到什么 ----
  printf("=== 表 3:lane 的三重身份(A 行 / B 列 / C 位置)===\n");
  printf("lane   A 行   B 列   C 行(4 个)        C 列\n");
  for (int l = 0; l < WAVE; ++l) {
    if (l != 0 && l != 1 && l != 15 && l != 16 && l != 17 && l != 48 && l != 63) continue;
    int lo = l & 15, hi = l >> 4;
    printf("%4d %6d %6d   %2d,%2d,%2d,%2d %12d\n", l, lo, lo,
           hi * 4, hi * 4 + 1, hi * 4 + 2, hi * 4 + 3, lo);
  }
  printf("  (只列出有代表性的几条)\n");
  printf("\n看点:lane 16 和 lane 0 的 A 行同为 0、C 列同为 0,但 C 行组不同;\n");
  printf("      lane 供给的 A 行号 != 它持有的 C 行号,这是最容易错的地方。\n\n");

  // ---- 可选:拆解单个元素 ----
  if (argc > 1) {
    int idx = atoi(argv[1]);
    if (idx < 0 || idx >= M * N) { printf("下标越界\n"); return 1; }
    int m = idx / N, n = idx % N;
    int lane = hP[idx] / 10, e = hP[idx] % 10;
    printf("=== 拆解 C[%d][%d] ===\n", m, n);
    printf("由 lane %d 的 acc[%d] 持有\n", lane, e);
    printf("  验算: 行 = (lane>>4)*4 + e = %d*4 + %d = %d  %s\n",
           lane >> 4, e, (lane >> 4) * 4 + e, ((lane >> 4) * 4 + e == m) ? "OK" : "??");
    printf("        列 = lane&15 = %d  %s\n", lane & 15,
           ((lane & 15) == n) ? "OK" : "??");
    printf("\n数据来源: A[%d][0..255] 点乘 B[%d][0..255]\n", m, n);
    printf("分 16 条 MMA 完成。每条指令里,这个元素需要的 A 行 %d\n", m);
    printf("由片段 row=%d 的那几条 lane 提供,即 lane %d / %d / %d / %d\n",
           m & 15, m & 15, (m & 15) + 16, (m & 15) + 32, (m & 15) + 48);
    printf("(4 个 k 组各一条)。注意这几条 lane 和持有结果的 lane %d 不是同一批。\n\n",
           lane);

    int32_t ref = 0;
    for (int k = 0; k < K; ++k) ref += (int32_t)hA[m * K + k] * (int32_t)hB[n * K + k];
    printf("CPU 参考 = %d,GPU = %d  -> %s\n", ref, hC[idx],
           ref == hC[idx] ? "相等" : "不等");

    printf("\n每条 MMA 指令覆盖的逻辑 k(置换后):\n");
    for (int i = 0; i < 4; ++i)
      for (int w = 0; w < 4; ++w) {
        printf("  指令 %2d (i=%d,w=%d): k = ", i * 4 + w, i, w);
        for (int qq = 0; qq < 4; ++qq) {
          int base = qq * 16 + i * 64 + w * 4;
          printf("%d-%d ", base, base + 3);
        }
        printf("\n");
      }
    printf("16 条指令的 k 集合合起来恰好覆盖 0..255 各一次。\n");
  } else {
    printf("提示: ./mma_teach <0..255> 可以拆解某个 C 元素,例如 ./mma_teach 37\n");
  }
  return 0;
}
