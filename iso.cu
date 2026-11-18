#include <mc_runtime.h>
#include <stdint.h>
typedef __attribute__((ext_vector_type(4))) int i32x4;
#define SLT 40
#define CALL(dst, src, k, K) __builtin_mxc_ldg_b128_bsm_predicator( \
    (dst), (void *)(src), 0, true, true, false, true, (k), (K), SLT)

// V1 动态 LDS + 丢弃返回值（和真实 kernel 一样）
__global__ void v1(const int8_t *A, int K, int32_t *o) {
  extern __shared__ int8_t s[];
  int t = threadIdx.x;
  CALL(s + t * 16, &A[t * 16], t * 16, K);
  __builtin_mxc_arrive(64);
  __builtin_mxc_barrier_inst();
  o[t] = *(const int32_t *)&s[t * 16];
}
// V2 动态 LDS + 接住返回值
__global__ void v2(const int8_t *A, int K, int32_t *o, int32_t *r) {
  extern __shared__ int8_t s[];
  int t = threadIdx.x;
  i32x4 v = CALL(s + t * 16, &A[t * 16], t * 16, K);
  __builtin_mxc_arrive(64);
  __builtin_mxc_barrier_inst();
  o[t] = *(const int32_t *)&s[t * 16];
  r[t] = v[0];
}
// V3 静态 LDS + 丢弃返回值
__global__ void v3(const int8_t *A, int K, int32_t *o) {
  __shared__ int8_t s[1024];
  int t = threadIdx.x;
  CALL(s + t * 16, &A[t * 16], t * 16, K);
  __builtin_mxc_arrive(64);
  __builtin_mxc_barrier_inst();
  o[t] = *(const int32_t *)&s[t * 16];
}
// V4 静态 LDS + 先往 LDS 写哨兵再异步搬（和探针里一样）
__global__ void v4(const int8_t *A, int K, int32_t *o) {
  __shared__ int8_t s[1024];
  int t = threadIdx.x;
  *(int32_t *)&s[t * 16] = 0x7E7E7E7E;
  __syncthreads();
  CALL(s + t * 16, &A[t * 16], t * 16, K);
  __builtin_mxc_arrive(64);
  __builtin_mxc_barrier_inst();
  o[t] = *(const int32_t *)&s[t * 16];
}
