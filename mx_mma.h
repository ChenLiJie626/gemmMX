// =============================================================================
// mx_mma.h  --  沐曦 C500 (MACA) 平台相关原语的唯一抽象层
//
// 全仓库只有这个文件直接碰 builtin。所有"我猜不到、必须在真机上核对"的东西
// 都集中在这里,核对方法写在每个宏上面。
//
// 核对命令(在装了 MACA SDK 的机器上跑):
//   grep -rn "mma_f32_16x16x16"  $MACA_PATH/include  | head
//   grep -rn "bsm"               $MACA_PATH/include  | head
//   mxcc -x maca --help | grep -i mma
// 若名字不同,只改本文件的 MX_MMA_BUILTIN / MX_BSM_* 宏即可,kernel 不用动。
// =============================================================================
#pragma once

#include <stdint.h>

// -----------------------------------------------------------------------------
// 0. 运行时 API:cucc 会把 cuda* 映射到 mc*;若直接用 mxcc 原生模式,打开下面分支
// -----------------------------------------------------------------------------
#if defined(MX_NATIVE_MACA)
#include <mc_runtime.h>
#define cudaMalloc            mcMalloc
#define cudaFree              mcFree
#define cudaMemcpy            mcMemcpy
#define cudaMemcpyHostToDevice mcMemcpyHostToDevice
#define cudaMemcpyDeviceToHost mcMemcpyDeviceToHost
#define cudaDeviceSynchronize mcDeviceSynchronize
#define cudaGetLastError      mcGetLastError
#define cudaGetErrorString    mcGetErrorString
#define cudaError_t           mcError_t
#define cudaSuccess           mcSuccess
#define cudaEvent_t           mcEvent_t
#define cudaEventCreate       mcEventCreate
#define cudaEventRecord       mcEventRecord
#define cudaEventSynchronize  mcEventSynchronize
#define cudaEventElapsedTime  mcEventElapsedTime
#define cudaEventDestroy      mcEventDestroy
#define cudaMemset            mcMemset
#else
#include <cuda_runtime.h>
#endif

#define MX_CHECK(x)                                                            \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[%s:%d] %s -> %s\n", __FILE__, __LINE__, #x,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------------
// 1. wave 宽度
//    C500 与多数类 CDNA 架构一样是 64 lane/wave。若实测是 32,编译时 -DMX_WAVE=32,
//    同时 MMA 的片段布局也要跟着换(见第 4 节)。
//    真机确认:kernel 里打印内置的 warpSize,或 grep SDK 里的 wavefrontSize。
// -----------------------------------------------------------------------------
#ifndef MX_WAVE
#define MX_WAVE 64
#endif

// -----------------------------------------------------------------------------
// 2. 半精度类型
//    主机侧只用 uint16_t 存储 + 软件转换,避免依赖任何 fp16 头文件;
//    设备侧用编译器原生 _Float16,这样能直接喂给 MMA builtin 的向量类型。
// -----------------------------------------------------------------------------
typedef uint16_t mx_h_store;  // host 侧存储
#if defined(__MACACC__) || defined(__MACA__) || defined(__CUDACC__)
typedef _Float16 mx_h;        // device 侧计算
typedef __attribute__((ext_vector_type(4))) _Float16 mx_h4;   // 4×fp16 = 8B
typedef __attribute__((ext_vector_type(4))) float     mx_f4;  // 4×fp32 = 16B
typedef __attribute__((ext_vector_type(8))) _Float16 mx_h8;   // 8×fp16 = 16B
#endif

// host 侧 fp32 <-> fp16 位操作转换(IEEE754 binary16,带 round-to-nearest-even)
static inline mx_h_store mx_f2h(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t man  = x & 0x7FFFFFu;
  if (exp <= 0) {  // 次正规/下溢
    if (exp < -10) return (mx_h_store)sign;
    man |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = man >> shift;
    if ((man >> (shift - 1)) & 1) h += 1;
    return (mx_h_store)(sign | h);
  }
  if (exp >= 31) return (mx_h_store)(sign | 0x7C00u);  // 上溢 -> inf
  uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
  if ((man & 0x1FFFu) > 0x1000u ||
      ((man & 0x1FFFu) == 0x1000u && ((man >> 13) & 1)))
    h += 1;
  return (mx_h_store)h;
}

static inline float mx_h2f(mx_h_store h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t man  = h & 0x3FFu;
  uint32_t x;
  if (exp == 0) {
    if (man == 0) { x = sign; }
    else {  // 次正规,规格化
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; exp--; }
      man &= 0x3FFu;
      x = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000u | (man << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &x, 4);
  return f;
}

#if defined(__MACACC__) || defined(__MACA__) || defined(__CUDACC__)

// -----------------------------------------------------------------------------
// 3. MMA 指令:C(16x16,fp32) += A(16x16,fp16) * B(16x16,fp16)
//
//    MX_HAS_MMA=0 时整个 MMA kernel 不参与编译(harness 会自动跳过),
//    这样不会出现"编译过了但算错"的静默失败。
// -----------------------------------------------------------------------------
#ifndef MX_HAS_MMA
#define MX_HAS_MMA 0
#endif

#ifndef MX_MMA_BUILTIN
// <<< 真机上用上面的 grep 命令核对这一行 >>>
#define MX_MMA_BUILTIN __builtin_mxc_mma_f32_16x16x16f16
#endif

__device__ __forceinline__ mx_f4 mx_mma_16x16x16(mx_h4 a, mx_h4 b, mx_f4 c) {
#if MX_HAS_MMA
  // 末三个 0 是 cbsz/abid/blgp 一类的广播控制位,单块 MMA 全填 0。
  // 若本机 builtin 只有 3 个参数,删掉这三个 0。
  return MX_MMA_BUILTIN(a, b, c, 0, 0, 0);
#else
  (void)a; (void)b;
  return c;  // 占位:MX_HAS_MMA=0 时此 kernel 不会被调用
#endif
}

// -----------------------------------------------------------------------------
// 4. MMA 寄存器片段布局 —— 本文件最需要在真机上验证的部分
//
//    下面是"候选布局",用 mma_probe (见 gemm_c500.cu --probe) 一跑就知道对不对。
//
//    MX_FRAG_LAYOUT == 0 (类 CDNA MFMA 16x16x16,64 lane):
//       A 片段: lane 持有 A[m][k0..k0+3],  m = lane%16,  k0 = (lane/16)*4
//       B 片段: lane 持有 B[k0..k0+3][n],  n = lane%16,  k0 = (lane/16)*4
//       C 片段: lane 持有 C[m0..m0+3][n],  n = lane%16,  m0 = (lane/16)*4
//
//    MX_FRAG_LAYOUT == 1: 同上,但 C 的 m/n 互换(有些实现 C 是行连续)
//       C 片段: lane 持有 C[m][n0..n0+3],  m = lane%16,  n0 = (lane/16)*4
//
//    关键性质(布局 0):A 和 B 的 lane→(k) 映射完全相同,所以 A/B 的共享内存
//    布局可以写成同一个函数,只是一个按 m 一个按 n 索引 —— 这是下面能用同一套
//    swizzle 的原因。
// -----------------------------------------------------------------------------
#ifndef MX_FRAG_LAYOUT
#define MX_FRAG_LAYOUT 0
#endif

// A 片段:该 lane 负责的行 m 的偏移
__device__ __forceinline__ int mx_frag_a_row(int lane) { return lane & 15; }
// A/B 片段:该 lane 负责的 k 起点(4 个连续 k)
__device__ __forceinline__ int mx_frag_k(int lane)     { return (lane >> 4) * 4; }
// B 片段:该 lane 负责的列 n 的偏移
__device__ __forceinline__ int mx_frag_b_col(int lane) { return lane & 15; }

// C 片段:第 i 个元素(i=0..3)对应 tile 内的 (m, n)
#if MX_FRAG_LAYOUT == 0
__device__ __forceinline__ int mx_frag_c_m(int lane, int i) { return (lane >> 4) * 4 + i; }
__device__ __forceinline__ int mx_frag_c_n(int lane, int i) { (void)i; return lane & 15; }
#else
__device__ __forceinline__ int mx_frag_c_m(int lane, int i) { (void)i; return lane & 15; }
__device__ __forceinline__ int mx_frag_c_n(int lane, int i) { return (lane >> 4) * 4 + i; }
#endif

// -----------------------------------------------------------------------------
// 5. BSM(block shared memory)原语
//
//    (a) 异步拷贝 global -> bsm,绕过寄存器。类似 NV 的 cp.async / AMD 的
//        global_load_lds。这是多 stage 流水的关键:发射后不占寄存器、不阻塞。
//    (b) MX_BSM_WAIT(n):等到"未完成的异步拷贝组"不超过 n 组。
//
//    MX_HAS_BSM_ASYNC=0 时自动退化成"global->寄存器->bsm"的两段式搬运,
//    功能完全一样,只是多占寄存器、流水稍差。先用退化版跑通,再开异步。
// -----------------------------------------------------------------------------
#ifndef MX_HAS_BSM_ASYNC
#define MX_HAS_BSM_ASYNC 0
#endif

#if MX_HAS_BSM_ASYNC
// <<< 真机核对:grep -rn "bsm_load\|async_copy" $MACA_PATH/include >>>
#define MX_BSM_COPY8(dst_bsm, src_glb)  __builtin_mxc_bsm_load_dwordx2((dst_bsm), (src_glb))
#define MX_BSM_COMMIT()                 __builtin_mxc_bsm_commit_group()
#define MX_BSM_WAIT(n)                  __builtin_mxc_bsm_wait_group((n))
#else
#define MX_BSM_COPY8(dst_bsm, src_glb)                                         \
  do { *(mx_h4 *)(dst_bsm) = *(const mx_h4 *)(src_glb); } while (0)
#define MX_BSM_COMMIT()  do {} while (0)
#define MX_BSM_WAIT(n)   do { (void)(n); } while (0)
#endif

// 显式的 8B / 16B 共享内存读写(让编译器一定发 ds_read_b64 / b128 而不是拆成多条)
__device__ __forceinline__ mx_h4 mx_lds_read8(const mx_h *p) {
  return *(const mx_h4 *)p;
}
__device__ __forceinline__ void mx_lds_write8(mx_h *p, mx_h4 v) {
  *(mx_h4 *)p = v;
}

#endif  // device 编译
