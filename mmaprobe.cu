// 把 __builtin_mxc_mma_16x16x16i8 扒到汇编
#include <mc_runtime.h>
#include <stdint.h>
typedef __attribute__((ext_vector_type(4))) int i32x4;

// ① 单条
__global__ void mma_one(const int *a, const int *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc = {0, 0, 0, 0};
  acc = __builtin_mxc_mma_16x16x16i8(a[t], b[t], acc);
  c[t] = acc;
}
// ② 四条串成一条依赖链（都累到同一个 acc）
__global__ void mma_chain4(const int *a, const int *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    acc = __builtin_mxc_mma_16x16x16i8(a[t * 4 + i], b[t * 4 + i], acc);
  c[t] = acc;
}
// ③ 四条互相独立（四个不同的 acc）
__global__ void mma_indep4(const int *a, const int *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] = (i32x4){0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    acc[i] = __builtin_mxc_mma_16x16x16i8(a[t * 4 + i], b[t * 4 + i], acc[i]);
#pragma unroll
  for (int i = 0; i < 4; ++i) c[t * 4 + i] = acc[i];
}
// ④ 累加器不清零，从内存读进来 —— 看它是不是真的就是普通寄存器
__global__ void mma_accin(const int *a, const int *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc = c[t];
  acc = __builtin_mxc_mma_16x16x16i8(a[t], b[t], acc);
  c[t] = acc;
}
