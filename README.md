# tutorial-int8-mma —— C500 int8 MMA 数据排布教程

这个分支不做实现之间的性能横评，只回答一个问题：

> **`__builtin_mxc_mma_16x16x16i8` 这条指令，数据到底是怎么排的？**
> 哪条 lane 交进去什么、拿回来什么，算完之后写回大矩阵的哪一格。

排布讲清楚之后，顺着同一套下标还能看出一处 LDS bank 冲突，
所以分支里也带了消冲突的 XOR swizzle 版本和它的图解。

从 `main` 基线分出，与 [`claude`](https://github.com/ChenLiJie626/gemmMX/tree/claude)、
[`codex`](https://github.com/ChenLiJie626/gemmMX/tree/codex) 两个实现分支平行。

---

## 文档

| 文件 | 说明 |
|---|---|
| [`docs/C500-int8-MMA-教程.html`](docs/) | **完整版**，9 节 15 张图。独立 HTML，双击即可打开（含交互查询表） |
| [`docs/C500-int8-MMA-教程.pdf`](docs/) | 同上的打印版，30 页 |
| [`docs/C500-int8-MMA-映射速查.html`](docs/) | **精简版**，只留讲映射关系的 8 张图 + 下标字典 + 四层偏移表 |
| [`docs/C500-int8-MMA-映射速查.pdf`](docs/) | 同上，12 页 |
| [`docs/C500-int8-MMA-XOR-swizzle.html`](docs/) | **XOR swizzle 消 bank 冲突**，5 张图。读指令 16 拍 → 8 拍 |
| [`docs/C500-int8-MMA-XOR-swizzle.pdf`](docs/) | 同上，11 页 |

HTML 是自包含的（SVG 内联、样式内联），只有字体走 Google Fonts CDN，离线打开会回落到系统字体。
浏览器里 Ctrl+P 就能印出与仓库里 PDF 一致的版面。

完整版目录：

```
01 这条指令算什么              06 交互查询：任给 lane 看它的三套坐标
02 数据排布（图 1–6）          07 输出来源
03 一条 lane 的一生            08 k 置换 k = 64c + 16q + 4w + j
04 区域划分 + 归位 + 写几次     09 完整 kernel
05 三段路 + 对照 kernel_test_v1.cu
```

精简版只保留八张讲「谁对应谁」的图，按数据流排：三张座位表 → 交进去 ≠ 拿回来 →
一个输出格子是谁凑的 → LDS 读哪儿 → 复用网格 → 定位图 → 一条 lane 的 64 个落点 → 整条链。

XOR swizzle 那份是性能向的续篇：教程讲清布局之后，这份指出同一个布局里
**取片段的 32 条 LDS 读每条要 16 拍，而下限是 8 拍**，原因是行距 256B 恰好是 bank
阵列宽度的整数倍；改法是一个异或 `槽位 = j ^ (r & 7)`。五张图依次是
bank 的几何 → 冲突现场 → XOR 表 → 修好之后 → 写路径没被破坏。

## 代码

全部用 `mxcc` 单文件编译，不依赖本仓库其它文件：

```bash
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
$MXCC -x maca -O3 mma_teach_256.cu -o mma_teach_256
```

| 文件 | 规模 | 作用 |
|---|---|---|
| `mma_teach.cu` | 1 wave，C 只有 16×16 | 最小教学版：没有任何分块，一眼看完一条 MMA |
| **`mma_teach_256.cu`** | 256×256×256 | **教程主线**。布局与 `kernel_test_v1.cu` 逐字一致，只把异步拷贝/屏障换成最朴素写法。内置三个探针 |
| `tutorial_i8_256.cu` | tile 256×256×256 | 性能线 v1，附 `prov` 来源探针 |
| `tutorial_i8_256_v2.cu` | 同上 | v2：B 走寄存器内 4×4 转置，BK 32→64 消除 bank 冲突 |
| `tutorial_i8_256_v3.cu` | 同上 | v3：v2 + 全局→寄存器预取 |
| `kernel_test_v1.cu` | tile 128×128×256 | 对照用的工程版（异步拷贝 + `arrive_gvmcnt`）。**需要 MACA SDK 的 `mctlass/arch/maca_memory.h`**，不在本仓库内 |
| `mma_swizzle_256.cu` | 256×256×256 | **XOR swizzle 版**。原始布局与 XOR 布局是同一个 `template <bool SW>` 的两个实例，除寻址宏外逐字相同，所以 `bench` 是严格同条件对照 |

数据约定（七个文件一致）：

```
A: M × K  行主序 int8
B: N × K  行主序 int8     <-- 注意是 [n][k]，host 侧已转置
C: M × N  行主序 int32
```

### 探针

`mma_teach_256.cu` 里三个探针 kernel 与主 kernel 结构逐行相同，只多一句记录，
教程里所有「真机 dump」都出自它们：

```bash
./mma_teach_256                  # 正确性 + 四层来源表
./mma_teach_256 164 81           # 拆解 C[164][81] 的完整来源链      (prov_256)
./mma_teach_256 reads 1 17       # 某条 lane 的 16 条 LDS 读坐标     (reads_256)
./mma_teach_256 partial 164 81   # 某个 C 元素的 16 条部分和逐条累加 (partial_256)
./mma_teach_256 bench [M N K]    # 计时
```

`mma_swizzle_256.cu` 还带一个**不需要 GPU** 的 bank 模型 —— 把 kernel 里那两条寻址宏
原样搬到 host 上数格子，并穷举检查 swizzle 的双射性：

```bash
./mma_swizzle_256 banks              # bank 深度直方图 + 双射检查（纯 CPU）
./mma_swizzle_256                    # 两个布局各对一次 CPU 参考
./mma_swizzle_256 bench 4096 4096 4096   # 两个布局的 us / TOPS 并排
```

## 验证状态

`mma_teach_256.cu` 在 C500（xcore1000，MACA 3.3.0）上跑过：
256³ 对 CPU 参考 **PASS，0 / 65536 不符**；闭式归属公式与 `prov_256` 的 dump 逐元素一致；
教程第 05 节的 LDS 读图、`C[164][81]` 与 `C[5][3]` 的逐条部分和都是 `reads_256` / `partial_256`
的真机输出，不是推导值。

`tutorial_i8_256*` 三版之间的吞吐对比不在此分支给结论。

`mma_swizzle_256.cu` 也在同一台机器上跑过了，两个布局对照：

| 尺寸 | 原始布局 | XOR 布局 | 加速 | 正确性 |
|---|---|---|---|---|
| 256³ | — | — | — | 两者 PASS 0/65536 |
| 1024³ | 44.6 us / 48.1 TOPS | 26.0 us / 82.7 TOPS | **1.72×** | 两者 PASS 0/1048576 |
| 2048³ | 256.7 us / 66.9 TOPS | 143.8 us / 119.5 TOPS | **1.79×** | — |
| 4096³ | 1653.9 us / 83.1 TOPS | 906.2 us / 151.7 TOPS | **1.83×** | — |

`bench` 会把两个布局**各跑两轮并交换先后顺序**，4096³ 下两轮差值小于 0.2%，
所以这不是时钟爬坡造成的假象。

证据分级：

| 结论 | 凭什么 | 状态 |
|---|---|---|
| 读指令 16 拍 → 8 拍，写指令仍在下限 | 地址公式的确定性推论，`banks` 穷举全部下标组合 | **已验** |
| swizzle 是双射（128 行 × 16 块，无越界无重叠） | 同上 | **已验** |
| 额外 LDS = 0 字节 | 行内重排，行距不变 | **已验** |
| 数值正确性 | C500 实跑，256³ / 1024³ 两个布局都 PASS | **已验** |
| 端到端 1.83× | C500 实跑 4096³，交换顺序两轮 | **已验** |
| 「原本约九成时间耗在 LDS 读上」 | 由实测加速比与 2.00× 理论上限**反推**，非 profiler 读数 | 推论 |
