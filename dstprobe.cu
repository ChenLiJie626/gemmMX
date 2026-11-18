// BSM 的落点到底是怎么给硬件的？把落点按 lane 打乱试试。
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define SLT 40
#define CK(x) do { mcError_t e=(x); if(e!=mcSuccess){printf("ERR %d %s\n",__LINE__,mcGetErrorString(e));exit(1);} } while(0)
// DST=0 落点按线程号连续；DST=1 落点在组内异或打乱
template <int DST>
__global__ void probe(const int8_t *A, int K, int32_t *out) {
  extern __shared__ int8_t s[];
  const int t = threadIdx.x;
  const int slot = DST ? ((t & ~3) | ((t & 3) ^ 1)) : t;
  __builtin_mxc_ldg_b128_bsm_predicator(
      s + slot * 16, (void *)&A[t * 16], 0, true, true, false, true, t * 16, K, SLT);
  __builtin_mxc_arrive(64);
  __builtin_mxc_barrier_inst();
  out[t] = *(const int32_t *)&s[t * 16];
}
template <int DST>
static void run(const char *n, const int8_t *dA, int32_t *dO, int32_t *hO) {
  CK(mcMemset(dO, 0, 64 * 4));
  probe<DST><<<1, 64, 1024>>>(dA, 4096, dO);
  CK(mcDeviceSynchronize());
  CK(mcMemcpy(hO, dO, 64 * 4, mcMemcpyDeviceToHost));
  printf("  %-26s LDS 槽位 0..7 读到：", n);
  for (int i = 0; i < 8; ++i) printf(" %d", hO[i]);
  int ok = 1;
  for (int i = 0; i < 64; ++i) if (hO[i] != 1000 + i) ok = 0;
  printf("   %s\n", ok ? "= 1000+槽位（原样落位）" : "<- 和槽位对不上");
}
int main(void) {
  int8_t *hA = (int8_t *)calloc(1024, 1);
  for (int i = 0; i < 64; ++i) *(int32_t *)&hA[i * 16] = 1000 + i;
  int8_t *dA; int32_t *dO; int32_t hO[64];
  CK(mcMalloc((void **)&dA, 1024)); CK(mcMalloc((void **)&dO, 256));
  CK(mcMemcpy(dA, hA, 1024, mcMemcpyHostToDevice));
  printf("A 的第 i 个 16B 块头 4 字节 = 1000+i；lane i 取第 i 块\n");
  run<0>("落点连续（我们的写法）", dA, dO, hO);
  run<1>("落点按 lane 打乱", dA, dO, hO);
  return 0;
}
