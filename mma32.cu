#include <mc_runtime.h>
#include <stdint.h>
typedef __attribute__((ext_vector_type(4))) int i32x4;
typedef __attribute__((ext_vector_type(2))) int i32x2;
__global__ void k32_one(const i32x2 *a, const i32x2 *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc = {0, 0, 0, 0};
  acc = __builtin_mxc_mma_16x16x32i8(a[t], b[t], acc);
  c[t] = acc;
}
__global__ void k32_chain4(const i32x2 *a, const i32x2 *b, i32x4 *c) {
  const int t = threadIdx.x;
  i32x4 acc = {0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    acc = __builtin_mxc_mma_16x16x32i8(a[t * 4 + i], b[t * 4 + i], acc);
  c[t] = acc;
}
