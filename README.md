# 沐曦 C500 fp16 GEMM 算子:逐段讲解

```
C = alpha * A * B + beta * C
A: M x K fp16    B: K x N fp16    C: M x N fp32
block tile: 128(M) x 128(N),K 方向宏 tile 256,stage 深度 BK = 32
线程: 256 = 4 wave,wave 排 2x2,每 wave 算 64x64
```

文件分工:

| 文件 | 内容 |
|---|---|
| `mx_mma.h` | **所有平台相关的东西都只在这里**:MMA builtin、BSM builtin、wave 宽度、片段布局。换 SDK 版本只改这一个文件 |
| `gemm_c500.cu` | kernel(v0 基线 + 模板化的 v1/v2/v3/v4)+ 片段布局探针 + 校验与计时 |
| `Makefile` | 构建、看寄存器用量、导汇编、跑 probe/check/bench |

---

## 0. 先说结论:最终衡量什么指标

**主指标是算力利用率**(业界叫 MFU / achieved-vs-peak):

```
实测 TFLOPS = 2 * M * N * K / 耗时
算力利用率  = 实测 TFLOPS / C500 fp16 峰值 TFLOPS
```

`2*M*N*K` 里的 2 是一次乘 + 一次加。harness 直接打印这个数,峰值要你用 `--peak` 传进来 —— C500 的 fp16 峰值算力和 HBM 峰值带宽请从沐曦官方规格书取,我没有可靠来源,不替你填。

参考标尺(通用 GPU GEMM 经验值,不是 C500 的承诺):

- 50% 以下:还有明显结构性问题(bank conflict、流水没起来、指令发射被挡)
- 60–75%:调优到位的手写 kernel 的常见区间
- 80%+:接近厂商库水平

辅助指标,用来定位"卡在哪":

1. **HBM 带宽利用率** —— harness 按 block tile 复用后的最小访存量算有效带宽。GEMM 是计算密集型,这个数**应该很低**;如果它接近峰值,说明 tile 太小、A/B 复用不够,是访存墙而不是算力墙。
2. **算术强度** (FLOP/Byte) —— 128x128 tile 下约 `2*128*128/((128+128)*2) = 64 FLOP/Byte`。v0 只有 0.5,差 128 倍,这就是分块的全部意义。
3. **MMA 发射占比** —— 用沐曦 profiler 看 MMA 指令占总发射的比例。利用率上不去时,看是 LDS 指令挤占了发射槽,还是在等 barrier。
4. **寄存器用量 / 是否 spill** —— `make res`。一旦 spill 到 scratch,利用率会断崖式掉。

**对数据的要求**:`make check` 在 512³ 上和 CPU fp64 累加结果对比,阈值 2e-2 相对误差(fp16 输入 + fp32 累加,K 大时这是正常量级)。4096³ 上 CPU 太慢,改为和 v0 的 GPU 结果对数。

---

## 1. `mx_mma.h`:平台抽象层

### 1.1 为什么要有这一层

我**无法在本机核对 C500 的真实 builtin 名字和 MMA 寄存器片段布局**。与其把猜测散落在 kernel 里,不如全部收到一个文件,每个宏上面写清核对命令:

```bash
grep -rn "mma_f32_16x16x16" $MACA_PATH/include | head
grep -rn "bsm" $MACA_PATH/include | head
```

名字不对就只改那一行,kernel 一行都不用动。

### 1.2 `MX_HAS_MMA` 默认为 0

默认 0 时 MMA kernel **整个不参与编译**,harness 只跑 v0 并打印提示。这是故意的:避免出现"编译过了但结果是错的"这种静默失败。确认 builtin 后用 `make HAS_MMA=1`。

### 1.3 半精度类型

主机侧用 `uint16_t` + 手写的 IEEE754 binary16 转换(`mx_f2h` / `mx_h2f`),不依赖任何 fp16 头文件;设备侧用编译器原生 `_Float16` 和 `ext_vector_type` 向量类型,因为 MMA builtin 的入参就是这种向量:

```c
typedef __attribute__((ext_vector_type(4))) _Float16 mx_h4;  // 8B,一个 A/B 片段
typedef __attribute__((ext_vector_type(4))) float     mx_f4; // 16B,一个 C 片段
```

### 1.4 MMA 片段布局(最需要验证的部分)

`MMA 16x16x16` + 64 lane 的假设布局(类 CDNA MFMA):

```
A 片段: lane 持有 A[m][k0 .. k0+3],  m  = lane % 16,  k0 = (lane/16)*4
B 片段: lane 持有 B[k0 .. k0+3][n],  n  = lane % 16,  k0 = (lane/16)*4
C 片段: lane 持有 C[m0 .. m0+3][n],  n  = lane % 16,  m0 = (lane/16)*4
```

64 个 lane 分 4 组,每组覆盖 4 个 k,合起来正好 16 个 k。

两个关键推论,后面的设计全建立在它们上面:

- **A 和 B 的 `lane -> k` 映射完全相同**,所以 A、B 的共享内存可以用同一套布局函数和同一套 swizzle,一个按 `m` 索引、一个按 `n` 索引。
- **一个片段是 4 个连续的 k**,所以共享内存必须 **k 连续**,这样取一个片段就是一条 8B 的 LDS 读指令。如果共享内存存成 `[k][m]`,取片段要 4 条 2B 读,LDS 指令数翻 4 倍,直接把发射槽吃光。

**验证方式**:`make probe` 跑 `mma_probe` kernel —— 单个 wave 做一块 16x16x16,和 CPU 结果对数。对了就打印"布局假设正确";错了就换 `make LAYOUT=1`(C 的 m/n 互换版);两个都错就看它打出来的 `got` vs `want` 第 0 行反推真实映射,再改 `mx_frag_*` 这几个函数。**这一步必须先过,否则后面全是错的。**

### 1.5 BSM 原语

`MX_BSM_COPY8` 在 `MX_HAS_BSM_ASYNC=1` 时映射到异步拷贝 builtin(global 直通共享内存,不过寄存器,发射后不阻塞 —— 多 stage 流水的理想搬运方式);为 0 时退化成普通的 `load + store`,功能一致,只是多占寄存器、流水差一点。**先用退化版跑通,再开异步。**

---

## 2. 分块参数:每个数字是怎么定下来的

```c
#define BM 128   #define BN 128   #define BK 32   #define KTILE 256
#define WAVES_M 2  #define WAVES_N 2        // wave 2x2
#define THREADS 256                          // 4 wave x 64 lane
#define WM 64  #define WN 64                 // 每 wave 的 tile
#define TM 4   #define TN 4                   // 每 wave 4x4 个 MMA tile
```

**倒推的顺序是:寄存器预算 → wave tile → block tile。**

每个 MMA tile 的 C 片段占 4 个 fp32/lane,所以累加器 = `TM * TN * 4`:

| block tile | wave 排布 | wave tile | 累加器/lane | 结论 |
|---|---|---|---|---|
| 256x256 | 2x2 | 128x128 | **256** | 需要架构有独立的累加寄存器堆,否则必 spill |
| 256x256 | 2x4 | 128x64 | 128 | 稳,但不是 wave2x2 |
| **128x128** | **2x2** | **64x64** | **64** | 选这个:wave2x2,寄存器宽松,多 stage 放得下 |

64 个累加器 + 8 个片段寄存器 + 16 个预取寄存器 + 地址,总共约 100 个 VGPR,很安全。

**共享内存预算**:单 stage = `(128*32 + 128*32) * 2B = 16KB`。

| STAGES | LDS 占用 | 说明 |
|---|---|---|
| 1 | 16 KB | v1/v2,单缓冲 |
| 3 | 48 KB | v3 默认 |
| 4 | 64 KB | v4,刚好吃满 64KB |

如果 128x128 还用 `BK=256`(一次搬完整个 K 宏 tile),单 stage 就要 128KB,放不下 —— **这就是 `KTILE=256` 必须再切成 `BK=32` 的 8 个 stage 的原因**。`KTILE` 在本实现里是 K 方向的断言粒度和将来做 split-K 的切分单位;流水必须跨越它,所以主循环是展平的 `K/BK` 步,没有嵌套。

---

## 3. v0 `gemm_naive`:1by1 搬运基线

```c
float acc = 0.f;
for (int k = 0; k < K; ++k) acc += (float)A[m*K+k] * (float)B[k*N+n];
C[m*N+n] = alpha*acc + beta*C[m*N+n];
```

一个线程一个 C 元素,每次乘加都从 global 读 A、B 各 2 字节:**算术强度 = 2 FLOP / 4 Byte = 0.5**,被 HBM 带宽彻底锁死,利用率通常是百分之零点几。

它的价值是两个:**功能基线**(大规模下给优化版当参考答案),以及**让"访存量"这件事量化** —— 后面每一步优化本质上都在提高这个 0.5。

---

## 4. 共享内存布局与 swizzle:`sidx()`

这是整个 kernel 里信息密度最高的 12 行,也是"解 bank conflict"的全部。

```c
#define SWZ_MASK (BK/4 - 1)          // BK=32 -> 8 个 chunk -> mask = 7

template <int SWZ>
__device__ int sidx(int row, int k) {
  int chunk = k >> 2;                 // 8B chunk 号(4 个 fp16)
  int off   = k & 3;
  if (SWZ) chunk ^= ((row >> 1) & SWZ_MASK);
  return row * BK + chunk * 4 + off;
}
```

A 存成 `[m][k]`,B 存成 `[n][k]`,逻辑上都是 **k 连续**(理由见 1.4)。

### 4.1 裸布局为什么冲突

LDS 是 32 个 bank × 4 字节。裸 `[row][BK]` 的行跨距 = `32 * 2B = 64B = 16 dword`,对 32 取模**只有 {0, 16} 两个值**。而取一个 MMA 片段要 16 个 lane(`m = lane%16`)各读 8B,16 个 lane 全挤在 2 组 bank 上 → **8 路冲突**,LDS 读变成 8 倍耗时。

### 4.2 XOR swizzle 怎么解

用行号去 XOR chunk 号:`chunk' = chunk ^ ((row>>1) & 7)`。

为什么是 `row>>1` 而不是 `row`:行跨距对 bank 取模只贡献了行号的**最低 1 位**(偶行 → bank 0,奇行 → bank 16),这一位已经被用掉了;剩下的打散工作必须交给 XOR,它能提供 8 个取值。`2 × 8 = 16`,16 个 lane 每人 2 个 dword,**正好铺满 32 个 bank 各一次 —— 读侧零冲突**。

验算一遍读侧(`c` 固定,lane 0..15 对应 row 0..15,单位 dword):

```
row 0: 0  + (c^0)*2      row 8 : 0  + (c^4)*2
row 1: 16 + (c^0)*2      row 9 : 16 + (c^4)*2
row 2: 0  + (c^1)*2      row 10: 0  + (c^5)*2
...                       ...
→ 偶行占 {0,2,...,14} 的 8 个,奇行占 {16,...,30} 的 8 个,互不重叠
```

### 4.3 写侧也顺带零冲突

A 的 global→bsm:连续 4 个线程写同一行的 4 个 16B,拆成 8B 后算下来是:

```
row 0 (tid 0..3) : banks {0,1} {4,5} {8,9}  {12,13}
row 1 (tid 4..7) : banks {16,17}{20,21}{24,25}{28,29}
row 2 (tid 8..11): banks {2,3} {6,7} {10,11}{14,15}
row 3 (tid12..15): banks {18,19}{22,23}{26,27}{30,31}
→ 32 个 bank 各命中一次,写侧也是零冲突
```

**同一个公式同时解掉读和写,这是选 XOR 而不是 padding 的理由** —— padding 还要额外占 LDS,并且读写要分别调参。

> 注意:以上推导基于"32 bank × 4B、同 dword 内多 lane 访问不算冲突"的常规 LDS 模型。C500 的 bank 数/宽度若不同,`SWZ_MASK` 和 `row>>1` 的移位量要重算。**上机后务必用 profiler 的 LDS conflict 计数器确认一次**,这是我唯一不能替你验证的性能假设。

---

## 5. 搬运:怎么"调整 load 方式满足 MMA 计算需求"

### 5.1 A 的搬运(理想情况)

A 是 `M x K` 行主序,而共享内存要 `[m][k]` —— **k 在两边都是连续的,所以直接 16B 向量拷贝,不需要转置**。

```c
const int a_vec_per_row = BK / 8;                  // 每行 32 个 fp16 = 4 个 16B
const int a_row0        = tid / a_vec_per_row;     // 本线程负责的行
const int a_k0          = (tid % a_vec_per_row)*8; // k 起点
const int a_row_stride  = THREADS / a_vec_per_row; // 第二份跨 64 行
```

把 tile 看成 `128*32/8 = 512` 个 16B 单元,256 线程每人 2 个。**连续的 tid 落在同一行的相邻 16B 上**,所以每 4 个线程凑出 64B 连续的 global 访存,合并良好。

落 bsm 时拆成两条 8B 写,因为 swizzle 的粒度是 8B chunk,一个 16B 单元的两个 chunk 被 XOR 后不一定还相邻。

### 5.2 B 的搬运(必须转置,代价说清)

B 是 `K x N` 行主序,但共享内存要 `[n][k]` —— **这里有一次不可回避的转置**。

采用的办法是"一个线程包一列 n 的一段 k":

```c
const int b_n       = tid % BN;          // 连续 lane -> 连续的 n
const int b_k_base  = (tid / BN) * 16;   // 每线程 16 个 k = 4 个 chunk
```

- **global 侧**:同一条指令里 64 个 lane 读连续的 n,`64 × 2B = 128B` 连续 → 仍然合并,但一条指令只搬 128B(A 侧是 1KB),**指令数是 A 的 8 倍**,这是 B 行主序的固有代价。
- **bsm 侧**:每线程写 4 个 8B chunk,地址 = `n*32 + (c ^ sw(n))*4`,16 个 lane 的 n 连续 → 按 4.2 的结论**零冲突**。

> 想消掉那 8 倍指令数,就把 B 预转置成 `N x K` 后用 `make BT=1`,B 的搬运路径立刻和 A 完全对称(16B 向量拷贝)。转置是 `O(KN)`,相对 `O(MNK)` 可忽略 —— 这也是 cuBLAS 偏爱 TN 布局的原因。

### 5.3 两种搬运形态

- `g2r` + `r2s`:global→寄存器→bsm,两段分开。**分开是流水的前提** —— 可以先发射 `g2r`(几百拍延迟),然后去算,算完再 `r2s`。
- `g2s`:直通,v1/v2 用,global 延迟完全暴露。

---

## 6. 计算:wave 2x2 与寄存器分块

### 6.1 wave 2x2

```c
const int wave_m = (wave / WAVES_N) * WM;   // 4 个 wave 排成 2x2
const int wave_n = (wave % WAVES_N) * WN;   // 各自负责一个 64x64
```

为什么 2x2 而不是 1x4(一个 wave 一条 128x32 的横条):

一个 wave 要算 `WM x WN` 的输出,需要从 bsm 读 `WM + WN` 行数据。

- 2x2 → `64 + 64 = 128` 行
- 1x4 → `128 + 32 = 160` 行

**方形排布让周长最小,也就是 LDS 读取量最小**。这就是 wave2x2 的全部道理 —— 和 block tile 取方形是同一个道理,只是换了一层。

### 6.2 寄存器分块:8 次读换 16 条 MMA

```c
mx_h4 fa[TM], fb[TN];
for (i < TM) fa[i] = lds_read8(&sA[buf][sidx(wave_m + i*16 + frag_row, kbase)]);
for (j < TN) fb[j] = lds_read8(&sB[buf][sidx(wave_n + j*16 + frag_col, kbase)]);

for (i < TM) for (j < TN)
    acc[i][j] = mx_mma_16x16x16(fa[i], fb[j], acc[i][j]);
```

先把 4 个 A 片段和 4 个 B 片段读进寄存器,**再做 4×4=16 条 MMA**。每个片段被复用 4 次:

| 方案 | LDS 读 | MMA | 比例 |
|---|---|---|---|
| 不分块(每条 MMA 现读现用) | 32 | 16 | 2 : 1 |
| **4x4 寄存器分块** | **8** | **16** | **1 : 2** |

LDS 带宽压力降到 1/4。这是把 bank conflict 解完之后的第二层 LDS 优化 —— **解冲突是降低单次读的代价,寄存器分块是降低读的次数**,两者正交,都要做。

一个 stage 内 `BK/MMA_K = 2` 次 MMA-K,共 32 条 MMA。

---

## 7. 多 stage 流水

```c
const int nk = K / BK;

// 预热:灌入 STAGES-1 个 stage
for (int s = 0; s < STAGES-1; ++s) g2s(s*BK, s);
__syncthreads();

for (int kt = 0; kt < nk; ++kt) {
  const int cur  = kt % STAGES;
  const int nk_i = kt + STAGES - 1;        // 本轮预取的 stage

  if (PF && nk_i < nk) g2r(nk_i * BK);     // ① 先发射 global 读
  compute(cur);                            // ② 用 32 条 MMA 盖住它的延迟
  if (nk_i < nk) r2s(nk_i % STAGES);       // ③ 数据到了再落 bsm
  __syncthreads();
}
```

### 7.1 为什么这个顺序有效

HBM 延迟是几百拍,一条 MMA 是十几拍。①②③ 的顺序让 32 条 MMA 和 global 读**物理上重叠**:发射 `g2r` 后不等结果就去算,等 `r2s` 要用寄存器时数据早回来了。

`STAGES=1` 走的是另一条路径(搬→同步→算→同步),延迟完全暴露 —— 这就是 v2 和 v3 的差距来源。

### 7.2 为什么一次迭代只需要一个 `__syncthreads()`

这是容易写错的地方。第 `kt` 轮写入的缓冲是 `(kt + STAGES-1) % STAGES = (kt-1) % STAGES`,也就是**第 `kt-1` 轮读过的那一个**。而第 `kt-1` 轮末尾的 barrier 已经保证所有线程都算完了 `kt-1`,所以现在写它绝不会撞上还在读的人。

对 `STAGES = 2` 同样成立 —— 很多实现在这里放两个 barrier,其实只要一个。barrier 在 256 线程规模下不便宜,省一个是实打实的收益。

### 7.3 stage 数怎么选

更多 stage = 更深的延迟隐藏,但 LDS 占用线性上升,进而压低一个 CU 上的 block 数(occupancy)。48KB(3 stage)和 64KB(4 stage)在 64KB LDS 的 CU 上都只能驻留一个 block,所以 harness 同时跑 v3/v4 让你**实测**哪个好,而不是猜。如果想试"2 stage + 两个 block 驻留"的路线,在 `main` 里加一行 `gemm_mma<2,1,1>` 的实例即可(stage 数是模板参数,不是编译宏)。

---

## 8. store C

```c
for (i < TM) for (j < TN) for (e < 4) {
  int m = block_m + wave_m + i*16 + mx_frag_c_m(lane, e);
  int n = block_n + wave_n + j*16 + mx_frag_c_n(lane, e);
  float *p = C + (size_t)m*N + n;
  *p = alpha * acc[i][j][e] + beta * (*p);
}
```

布局 0 下同一 tile 内 16 个 lane 的 `n` 连续 → `16 × 4B = 64B` 合并写。4 个元素在 M 方向跨行,所以是 4 条独立的 4B 写,不能向量化。

`beta != 0` 时要读回 C,多一遍 `M*N*4` 的读。想省掉就为 `beta == 0` 单独出一个分支(常见优化,这里没做,因为它不影响主指标)。

**若 probe 显示是布局 1**(C 的 n 方向连续),这里可以合成一条 16B 写 —— 顺手的收益,改 4 行。

---

## 9. 上机顺序

```bash
# ① 先确认工具链和 arch 名字(HAS_MMA=0,只编基线)
make && ./gemm_c500 -m 512 -n 512 -k 512

# ② 核对 builtin 名字
grep -rn "mma_f32_16x16x16" $MACA_PATH/include | head

# ③ 验证片段布局 —— 这一步不过,后面全是错的
make HAS_MMA=1 probe
#   失败就 make HAS_MMA=1 LAYOUT=1 probe

# ④ 小规模对 CPU 验正确性
make HAS_MMA=1 check

# ⑤ 看寄存器用量,确认没有 spill
make HAS_MMA=1 res

# ⑥ 确认 MMA 真发出来了、LDS 读是 64 位宽而不是被拆成 b16
make HAS_MMA=1 asm

# ⑦ 性能与利用率(PEAK/BWPEAK 填规格书的数)
make HAS_MMA=1 bench PEAK=<fp16峰值TFLOPS> BWPEAK=<HBM峰值GB/s>

# ⑧ 逐项加码
make HAS_MMA=1 BT=1    bench ...   # B 预转置
make HAS_MMA=1 ASYNC=1 bench ...   # 异步 bsm 拷贝
```

## 10. 还没做、按收益排序的后续项

1. **block 调度 swizzle**:把 `blockIdx` 按 `group` 重排(比如每 8 个 block 走一列),提高 A/B 在 L2 的复用。通常 5–15%,改 3 行,是性价比最高的一项。
2. **异步 bsm 拷贝**:`ASYNC=1`,省掉中转寄存器,流水更紧。
3. **`beta == 0` 专用分支**:省一遍 C 的读。
4. **split-K**:`M`、`N` 小而 `K` 很大时,现在的 grid 铺不满所有 CU。按 `KTILE=256` 切分 + 原子加或二次归约。
5. **双缓冲到寄存器级**:在 `compute` 内部把下一个 k-step 的片段读提前到当前 MMA 之前,隐藏 LDS 延迟。
6. **MMA 变体**:若 SDK 提供 `32x32x8` 之类的大块 MMA,每条指令的计算/发射比更高,但片段布局和累加器数量都要重算。

---

## 11. 已知未验证项(需要你在 C500 上确认)

本机没有任何编译器(nvcc/mxcc/cucc/clang 都不存在),**这份代码没有经过编译,也没有跑过**。按风险排序:

1. **MMA / BSM builtin 的名字和参数个数** —— 必错即改,`mx_mma.h` 第 3、5 节,grep 命令已给。
2. **MMA 片段布局** —— `make probe` 判定,错了换 `LAYOUT=1`。
3. **wave 宽度是否为 64** —— 若是 32,`MX_WAVE=32`,且片段布局要整套重算(4 个 lane 组变 2 个)。
4. **LDS bank 模型** —— 影响 swizzle 是否真的零冲突,用 profiler 的冲突计数器确认。
5. **`--offload-arch` 的真实 arch 名** —— `Makefile` 里的 `mxgpu_c500` 是占位。
6. **语法/编译错误** —— 没编译过,第一遍大概率要修几处。
