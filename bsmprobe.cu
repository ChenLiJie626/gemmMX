// 查 __builtin_mxc_ldg_b128_bsm_predicator 的四个 i1 标志到底干什么。
// 返回值一律丢弃 —— 接住它会让编译器后端崩（见 iso.cu 的 v2）。
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SLT 40
#define arrive_gvmcnt(n) __builtin_mxc_arrive(64 + (n))
#define CK(x) do { mcError_t e=(x); if(e!=mcSuccess){ \
  printf("ERR %d %s\n",__LINE__,mcGetErrorString(e)); exit(1);} } while(0)
#define SENT 0x7E7E7E7E

template <bool F1, bool F2, bool F3, bool F4>
__global__ void probe(const int8_t *A, int K, int32_t *out) {
  extern __shared__ int8_t s[];
  const int tid = threadIdx.x;
  for (int i = 0; i < 4; ++i) *(int32_t *)&s[tid * 16 + i * 4] = SENT;
  __syncthreads();
  const int koff = tid * 16;
  __builtin_mxc_ldg_b128_bsm_predicator(
      s + tid * 16, (void *)&A[koff], 0, F1, F2, F3, F4, koff, K, SLT);
  arrive_gvmcnt(0);
  __builtin_mxc_barrier_inst();
  out[tid] = *(const int32_t *)&s[tid * 16];
}

static const char *tag(int32_t v, int lane) {
  if (v == SENT) return "哨兵(没动)";
  if (v == 0) return "零";
  if (v == 1000 + lane) return "数据";
  return "别的";
}

template <bool F1, bool F2, bool F3, bool F4>
static void run(const char *name, const int8_t *dA, int K, int32_t *dO,
                int32_t *hO) {
  CK(mcMemset(dO, 0x11, 64 * 4));
  probe<F1, F2, F3, F4><<<1, 64, 1024>>>(dA, K, dO);
  CK(mcDeviceSynchronize());
  mcError_t le = mcGetLastError();
  if (le != mcSuccess) { printf("  %-18s 启动失败 %s\n", name, mcGetErrorString(le)); return; }
  CK(mcMemcpy(hO, dO, 64 * 4, mcMemcpyDeviceToHost));
  int nd = 0, ns = 0, nz = 0, no = 0;
  for (int i = 0; i < 64; ++i) {
    if (hO[i] == 1000 + i) ++nd; else if (hO[i] == SENT) ++ns;
    else if (hO[i] == 0) ++nz; else ++no;
  }
  printf("  %-18s (%d,%d,%d,%d)  lane0=%-10s lane31=%-10s lane32=%-10s lane63=%-10s"
         "  | 数据 %2d 哨兵 %2d 零 %2d 其它 %2d\n",
         name, F1, F2, F3, F4, tag(hO[0], 0), tag(hO[31], 31),
         tag(hO[32], 32), tag(hO[63], 63), nd, ns, nz, no);
}

int main(void) {
  const int NCOL = 64 * 16;
  int8_t *hA = (int8_t *)malloc(NCOL);
  memset(hA, 0, NCOL);
  for (int lane = 0; lane < 64; ++lane)
    *(int32_t *)&hA[lane * 16] = 1000 + lane;
  int8_t *dA; int32_t *dO;
  CK(mcMalloc((void **)&dA, NCOL));
  CK(mcMalloc((void **)&dO, 64 * 4));
  CK(mcMemcpy(dA, hA, NCOL, mcMemcpyHostToDevice));
  int32_t hO[64];

  printf("A 的每 16 字节头 4 字节 = 1000+lane；LDS 预填哨兵 0x%08X\n", SENT);
  printf("标志顺序 = builtin 的第 4..7 个实参 (F1=Ret0, F2, F3, F4)\n\n");

  printf("=== K=512：lane 0-31 谓词真，lane 32-63 谓词假 ===\n");
  run<true,  true,  false, true >("我们写的", dA, 512, dO, hO);
  run<false, true,  false, false>("SDK 包装器", dA, 512, dO, hO);
  run<false, true,  false, true >("只把 F1 翻成 0", dA, 512, dO, hO);
  run<true,  true,  false, false>("只把 F4 翻成 0", dA, 512, dO, hO);
  run<true,  false, false, true >("只把 F2 翻成 0", dA, 512, dO, hO);
  run<true,  true,  true,  true >("只把 F3 翻成 1", dA, 512, dO, hO);

  printf("\n=== K=2048：谓词全真（这才是真实 kernel 里的情形）===\n");
  run<true,  true,  false, true >("我们写的", dA, 2048, dO, hO);
  run<false, true,  false, false>("SDK 包装器", dA, 2048, dO, hO);
  run<true,  false, false, true >("只把 F2 翻成 0", dA, 2048, dO, hO);
  run<true,  true,  true,  true >("只把 F3 翻成 1", dA, 2048, dO, hO);

  printf("\n=== K=0：谓词全假 ===\n");
  run<true,  true,  false, true >("我们写的", dA, 0, dO, hO);
  run<false, true,  false, false>("SDK 包装器", dA, 0, dO, hO);
  return 0;
}
