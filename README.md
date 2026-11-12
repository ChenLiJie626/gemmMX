# C500 GEMM — Codex 实现

原生MACA `.cu`，实现 `D = alpha * A * B + beta * C`。

- **tile N×M×K = 128×128×256**：每个CTA计算128×128输出，256线程，wave2×2。
- FP16 A/B，FP32 MMA累加与C/D；M/N为128的倍数，K为256的倍数。
- A行主序；B默认列主序、K连续；`--row-b`提供行主序B转换入口，并另计转换+GEMM时间。
- mode0–7：标量直接读取、标量BSM搬运、native BSM builtin、XOR布局、2/3/4-stage和寄存器预取流水线候选。

## 文件

| 文件 | 内容 |
|---|---|
| [c500_gemm.cu](c500_gemm.cu) | kernel、版本分发、FP64参考校验与GPU event基准 |
| [README_逐段讲解.md](README_逐段讲解.md) | 对照源码[01]–[16]逐段解释、布局推导、流水线、性能指标与验收方法 |
| [build.sh](build.sh) | MACA mxcc编译入口 |
| [run_checks.sh](run_checks.sh) | 多尺寸与特殊输入正确性检查 |
| [benchmark.sh](benchmark.sh) | 多尺寸、各版本基准 |
| [validate_layout.mjs](validate_layout.mjs) | 主机端布局和stage生命周期数学模型 |
| [验证状态.txt](验证状态.txt) | 已执行与未执行验证的边界 |
| [NOTICE.md](NOTICE.md) | 公开native接口依据与来源说明 |

## 使用

在装有C500 MACA SDK的Linux环境运行：

```bash
export MACA_PATH=/opt/maca
bash build.sh
bash run_checks.sh
bash benchmark.sh > benchmark.txt

# M N K MODE REPEATS ALPHA BETA
./c500_gemm 128 128 256 -1 20 1 1
./c500_gemm 4096 4096 4096 4 50 1 0
```

mode=-1运行全部版本；mode4为2-stage，mode5为3-stage。它们是待测候选，尚未选出真机最优配置。

## 验证状态

主机端布局数学模型与三个shell脚本语法检查通过。**当前环境没有MACA编译器或C500运行环境，尚未完成GPU编译、数值验证、bank conflict profiler分析或性能测量。** 主机模型通过不代表硬件builtin ABI和内存可见性已验证。

正确性通过后，以延迟、`TFLOPS=2MNK/(t_ms*10^9)`和相对mcBLAS效率比较性能；峰值利用率的分母须对应本卡dense FP16→FP32规格。未提供任何虚构性能数据。

用户提供的Claude实现独立保存在[claude分支](https://github.com/ChenLiJie626/gemmMX/tree/claude)。
