# tutorial-int8-mma —— C500 int8 MMA 数据排布教程

这个分支不比性能，只回答一个问题：

> **`__builtin_mxc_mma_16x16x16i8` 这条指令，数据到底是怎么排的？**
> 哪条 lane 交进去什么、拿回来什么，算完之后写回大矩阵的哪一格。

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

数据约定（六个文件一致）：

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

## 验证状态

`mma_teach_256.cu` 在 C500（xcore1000，MACA 3.3.0）上跑过：
256³ 对 CPU 参考 **PASS，0 / 65536 不符**；闭式归属公式与 `prov_256` 的 dump 逐元素一致；
教程第 05 节的 LDS 读图、`C[164][81]` 与 `C[5][3]` 的逐条部分和都是 `reads_256` / `partial_256`
的真机输出，不是推导值。

性能不是这个分支的议题，`tutorial_i8_256*` 三版的吞吐对比未在此处给出结论。
