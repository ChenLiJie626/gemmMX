# gemmMX — 沐曦 C500 GEMM 实现对比

矩阵乘加：`D = alpha * A * B + beta * C`（原地版本写回 C）。

约定 tile 顺序为 **N × M × K = 128 × 128 × 256**。实际 M、N 为128的倍数，K为256的倍数；FP16输入、FP32累加与输出。

本仓库通过两个独立分支保存实现，均从此公共基线分出：

| 分支 | 内容 |
|---|---|
| [claude](https://github.com/ChenLiJie626/gemmMX/tree/claude) | 用户提供的 Claude 实现：`gemm_c500.cu`、`mx_mma.h`、Makefile、原始说明 |
| [codex](https://github.com/ChenLiJie626/gemmMX/tree/codex) | Codex 实现：native MMA/BSM、XOR布局、1–4 stage、寄存器预取候选、校验与计时脚本、逐段讲解 |

```bash
git clone https://github.com/ChenLiJie626/gemmMX.git
cd gemmMX
git switch claude  # 或 git switch codex
```

当前发布的是源码与实验候选，**两个版本均未在本次环境完成 MACA 编译、C500 数值验证或性能验收**。Codex分支的主机布局模型与shell语法检查已通过；这不能替代GPU测试。

正确性通过后，再在相同数据布局、alpha/beta、尺寸及设备配置下比较延迟、TFLOPS、相对mcBLAS效率与dense FP16→FP32峰值利用率。具体构建方式和限制请阅读各分支说明。
