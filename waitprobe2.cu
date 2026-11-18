// 谁在挡？让 lane 去读「别人搬进来的」那一份，编译器推不出这个依赖。
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define SLT 40
#define CK(x) do { mcError_t e=(x); if(e!=mcSuccess){ \
  printf("ERR %d %s\n",__LINE__,mcGetErrorString(e)); exit(1);} } while(0)
#define SENT 0x5A5A5A5A
// WAIT: 0=不发 arrive  1=arrive(64)   BAR: 要不要 barrier   CROSS: 读别人的
template <int WAIT, int BAR, int CROSS>
__global__ void probe(const int8_t *A, int K, int32_t *bad, int32_t *sent) {
  extern __shared__ int8_t s[];
  const int tid = threadIdx.x;
  const long base = (long)blockIdx.x * 4096 * 256;
#pragma unroll
  for (int i = 0; i < 8; ++i) *(int32_t *)&s[(i * 256 + tid) * 16] = SENT;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int off = (i * 256 + tid) * 16;
    const long g = base + off;
    __builtin_mxc_ldg_b128_bsm_predicator(
        s + off, (void *)&A[g], 0, true, true, false, true, (int)(g % K), K, SLT);
  }
  if (WAIT) __builtin_mxc_arrive(64);
  if (BAR) __builtin_mxc_barrier_inst();
  // CROSS=1：读 (tid+64)%256 那份 —— 是别的 lane 搬的
  const int rt = CROSS ? ((tid + 64) & 255) : tid;
  int nb = 0, ns = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int off = (i * 256 + rt) * 16;
    const int32_t got = *(const int32_t *)&s[off];
    const int32_t want = (int32_t)(base + off);
    if (got == SENT) ++ns; else if (got != want) ++nb;
  }
  if (nb) atomicAdd(bad, nb);
  if (ns) atomicAdd(sent, ns);
}
template <int W, int B, int C>
static void run(const char *name, const int8_t *dA, int K, int32_t *d2, int grid) {
  int32_t h2[2] = {0, 0};
  CK(mcMemcpy(d2, h2, 8, mcMemcpyHostToDevice));
  probe<W, B, C><<<grid, 256, 32768>>>(dA, K, d2, d2 + 1);
  CK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("  %-34s 启动失败 %s\n", name, mcGetErrorString(le)); return; }
  CK(mcMemcpy(h2, d2, 8, mcMemcpyDeviceToHost));
  const long tot = (long)grid * 256 * 8;
  printf("  %-34s 脏 %8d / %ld  哨兵 %8d  -> %s\n", name, h2[0], tot, h2[1],
         (h2[0] || h2[1]) ? "**没等到**" : "全对");
}
int main(int argc, char **argv) {
  const int grid = argc > 1 ? atoi(argv[1]) : 512;
  const long N = (long)grid * 4096 * 256 + 32768;
  int8_t *hA = (int8_t *)malloc(N);
  for (long i = 0; i + 4 <= N; i += 16) *(int32_t *)&hA[i] = (int32_t)i;
  int8_t *dA; int32_t *d2;
  CK(mcMalloc((void **)&dA, N));
  CK(mcMalloc((void **)&d2, 8));
  CK(mcMemcpy(dA, hA, N, mcMemcpyHostToDevice));
  printf("grid=%d，每 block 搬 32KB\n\n", grid);
  printf("读自己搬的那一份：\n");
  run<0, 0, 0>("不 arrive 不 barrier", dA, 1 << 20, d2, grid);
  run<1, 0, 0>("arrive(64) 不 barrier", dA, 1 << 20, d2, grid);
  printf("\n读别人搬的那一份（编译器推不出依赖）：\n");
  run<0, 0, 1>("不 arrive 不 barrier", dA, 1 << 20, d2, grid);
  run<0, 1, 1>("只 barrier，不 arrive", dA, 1 << 20, d2, grid);
  run<1, 0, 1>("只 arrive(64)，不 barrier", dA, 1 << 20, d2, grid);
  run<1, 1, 1>("arrive(64) + barrier（我们的写法）", dA, 1 << 20, d2, grid);
  printf("\n再跑一遍确认不是偶然：\n");
  run<0, 0, 1>("不 arrive 不 barrier", dA, 1 << 20, d2, grid);
  run<0, 1, 1>("只 barrier，不 arrive", dA, 1 << 20, d2, grid);
  return 0;
}
