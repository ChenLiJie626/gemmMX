# 沐曦 C500：N×M×K = 128×128×256 矩阵乘加实现与逐段讲解

这里实现 `D = α A B + β C`。源码是原生 MACA `.cu`，直接使用 C500 的 MMA 和 global→BSM builtin，未使用 NVIDIA WMMA/PTX。

**验证状态：已完成源码、主机端布局数学模型和循环缓冲区生命周期检查；未完成 mxcc 编译、C500 数值验证、bank conflict 实测和最优参数选择。当前 Windows 环境没有检测到 MACA 编译器或 C500 执行环境。文中的优化版均为待实测候选，不能把模型通过当作 GPU 测试通过。**

## 1. 接口、数据与 tile 的准确含义

默认约定如下，用户未另行指定的数据类型按这一组实现：

| 项目 | 约定 |
|---|---|
| A | FP16，逻辑 M×K，行主序，`A[m*K+k]` |
| B | FP16，逻辑 K×N，列主序，`B[n*K+k]`；K 连续 |
| C、D | FP32，M×N，行主序，`C[m*N+n]` |
| 计算 | FP16×FP16，FP32 MMA 累加；最后应用 α、β |
| 尺寸 | M、N 是正的128的倍数，K是正的256的倍数；不要求 M=N=K |
| stride | 紧密存储；此版本没有任意 lda/ldb/ldc 接口 |
| α、β | 有限 FP32；β=0 时不读取 C |
| 其他 | 单卡、单 GPU 默认 stream；不实现 bias、激活、转置 A、BF16 |

“矩阵乘加”也可写成 `C←AB+C`。本例用独立 D 保存结果，避免基准中多次运行导致 C 反复累加；单次调用内核时 D 可以等于 C。A/B 与输出不能重叠。

如果 B 是常见的行主序 `Brow[k*N+n]`，使用 `--row-b`。程序执行转换内核得到上述 K 连续布局，并同时报告纯 GEMM 与“转换+GEMM”的时间。此转换是独立的布局准备步骤；**BSM builtin 本身不等于矩阵转置或自动生成 MMA layout**。

### 一个 CTA 对应128×128输出，逻辑 K tile为256

按修正后的 nmk tile：N=128、M=128、K=256。每个 CTA 负责一个128×128输出区域，归约循环沿K每256分一个逻辑tile。每个逻辑tile内含八个K=32分片；每个分片用两组K=16的native MMA完成。

不能把完整 FP16 A[128,256] 与 B[256,128] 都放入一个64 KiB BSM预算：两者合计128 KiB。这里始终只暂存较薄的K分片。`TileK=256`的逻辑归约步长与`SliceK=32`的驻留深度是不同参数。

所有版本都由一个物理CTA计算一个128×128输出tile，CTA包含2×2个wave：

| mode | 物理 CTA 输出 | CTA 内 wave 排列 | threads | 每 wave 输出 | BSM |
|---|---:|---:|---:|---:|---:|
| 0 | 128×128 | 2×2 | 256 | 64×64 | 0 |
| 1、2、3 | 128×128 | 2×2 | 256 | 64×64 | 16 KiB |
| 4 | 128×128 | 2×2 | 256 | 64×64 | 32 KiB |
| 5、7 | 128×128 | 2×2 | 256 | 64×64 | 48 KiB |
| 6 | 128×128 | 2×2 | 256 | 64×64 | 64 KiB |

grid直接设为`(N/128, M/128)`；没有跨CTA拼接输出tile，也不使用split-K或原子累加。

这里的 **wave 2×2 指四个 wave 沿 M/N 排列**。每个 wave 内部不是只有 2×2 条 MMA，而是持有 4×4 个分布式 accumulator fragment。如果你说的 wave2×2 是“单个 wave 只计算 2×2 个 MMA 子块”，那是另一种参数定义，需要改 wave 输出和 wave 数量。

128×128输出共16384个FP32结果，分给256个线程后，每线程64个FP32累加器，另需A/B、地址和流水线临时值。`__launch_bounds__(256)`告知编译器block大小，但寄存器数和spill仍需从资源报告确认。

## 2. 编译与运行

将整个目录复制到装有 C500 MACA SDK 的 Linux 主机，进入该目录：

```bash
export MACA_PATH=/opt/maca
bash build.sh
bash run_checks.sh
bash benchmark.sh > benchmark.txt
```

`build.sh` 使用 `mxcc -x maca -std=c++17 -O3`。两个 BSM 相关 LLVM 参数来自参考 native kernel 工程的编译设置；不同 SDK 版本可能改变参数或 builtin ABI。请先保留完整编译日志，不要自行改成 NVIDIA 的 `nvcc -arch=sm_80`。

可以指定编译器或追加该 SDK 支持的编译参数：

```bash
MXCC=/your/sdk/bin/mxcc bash build.sh
# 资源报告开关请先通过本机 mxcc --help 核对，再作为额外参数传给 build.sh。
```

命令行：

```text
./c500_gemm M N K MODE REPEATS ALPHA BETA [--row-b] [--peak=TFLOPS] [--pattern=...]
```

例如：

```bash
# 所有版本，最小尺寸，全量正确性检查
./c500_gemm 128 128 256 -1 20 1 1
# 128×128 CTA，2-stage
./c500_gemm 4096 4096 4096 4 50 1 0
# wave2×2，3-stage，B 来自行主序接口
./c500_gemm 2048 4096 1024 5 50 0.75 -0.25 --row-b
```

mode 对照：

| mode | 名称 | 此次主要改动 |
|---:|---|---|
| 0 | scalar-direct | A/B 标量读取并直接组装 native fragment；无 BSM |
| 1 | scalar-bsm | 标量 global→寄存器→BSM，1-stage，无 XOR |
| 2 | builtin-bsm | 128-bit global→BSM builtin，1-stage，无 XOR |
| 3 | xor-wave2x2 | XOR BSM 布局，四个 wave 计算 128×128 |
| 4 | stage2 | 两个 BSM stage |
| 5 | stage3 | 三个 BSM stage |
| 6 | stage4 | 四个 BSM stage |
| 7 | stage3-reg-prefetch | 三个 BSM stage，加当前/下一片寄存器 fragment |

这不是“版本号越大越快”的排序。小问题、长 K、矩形问题可能分别偏好不同版本。

## 3. 对照源码 [01]–[16] 逐段讲解

### [01] 头文件、向量类型、常量和错误检查

`maca.h`、`maca_fp16.h`、`mc_runtime.h` 提供 native 类型、FP16 转换和 GPU runtime。`.cu` 只是文件后缀；实际语言和 ABI 由 `mxcc -x maca` 决定。

`U2` 为两个 32 位整数，包含四个 FP16；`U4` 包含八个 FP16，共 16 字节；`F4` 是四个 FP32。这里必须用 `__NATIVE_VECTOR__`，因为 builtin 接收编译器的 native vector 类型。不能随意用一个看起来大小相同的 struct 替代。

`Wave=64`、`TileM=TileN=128`、`TileK=256`、`SliceK=32`明确wave、输出tile、逻辑归约tile与实际驻留深度。主机查询`warpSize`，若不是64就拒绝执行，防止错误的lane映射静默产生错误结果。

`MC_CHECK` 检查每次 runtime 调用。launch 后 `mcGetLastError` 检查启动错误，后续同步/拷贝检查异步执行错误。示例遇错退出，生产库可将此策略改成向调用方返回错误码。

### [02] 为什么每 lane 有 64 个累加结果

一个 wave 有 64 个 lane，负责 64×64=4096 个结果，因此平均每 lane 保存 64 个结果。

`Accumulators.c[4][4]` 包含 16 个 `F4`，正好 64 个 FP32。`Fragments.a[4]`、`b[4]` 分别保存四个 8-half 向量。它们在数学上代表不同输出行或列，而不是整个连续的 64×32 矩阵；整个矩阵分散在 64 个 lane 中。

`pack8` 用八个 16-bit 标量值拼成四个 32-bit word：

```text
word[j] = half_bits[2j] | (half_bits[2j+1] << 16)
```

这里只搬动 FP16 位模式，没有执行数值转换。“1by1 load”表示源码逐元素读取；优化编译器可能合并访问。因此若课程要求机器指令也严格逐标量，必须看反汇编。没有为强制低效而添加 volatile。

### [03] MMA 调用与两个 K 子集

`__builtin_mxc_mma_16x16x16f16` 每次让整个 wave 协作完成一块 16×16×16 的矩阵乘加，每 lane 接收两组 `U2` 和一个 `F4` 累加器。

本例 native operand 顺序是 `(b, a, acc)`。这是与后续 store 坐标配套的选择，不能按变量名直觉换成 `(a,b,acc)`。公开参考实现也使用这一顺序；推导和模型检查均依赖该 native 布局约定，最终仍须通过设备数值测试确认当前 SDK ABI。

一次 32-wide 分片中，每 lane 拿到 K 的 `8q..8q+7`。第一个 MMA 使用每 lane 的前四个值，覆盖：

```text
K 集合 0：0..3、8..11、16..19、24..27
K 集合 1：4..7、12..15、20..23、28..31
```

这两个集合并不是全局连续的两个 16-wide 区间，但 A/B 使用相同 K 排列，求和结果仍等于完整的 32 项内积。native MMA 接收的是 lane 分布后的矩阵，而不是一个随意的连续数组。

`mma_half` 的两层 4×4 循环更新不同的 accumulator，扩大指令级并行度。每个 wave 每个 K=32 分片调用 `2×4×4=32` 条 wave 级 MMA，数学工作量为 `32×2×16³=262144 FLOPs`，等于 `2×64×64×32`。

### [04] A/B 如何按 MMA 要求读取

设 `r=lane%16`，`q=lane/16`，则输入分配为：

```text
A：m = wave_m_origin + 4r+i，k = slice_origin + 8q+e
B：n = wave_n_origin + 4r+j，k = slice_origin + 8q+e
i,j ∈ [0,3]，e ∈ [0,7]
```

因此 64 个 lane 的四份 A fragment 正好覆盖 A[64,32]；B 同理覆盖 B[32,64]。B 在物理存储上 K 连续，所以读取一个列片段也能使用连续的 16 字节访问。

mode 0 完全通过 global memory 得到这些 fragment，作为最容易观察数据映射的基线。它没有利用同一 CTA 不同 wave 之间的 A/B 复用。

### [05] BSM 布局与 XOR

BSM 按 16 字节 vector 编址。四个矩阵行构成一个 group：

```text
group = row / 4
col   = 4*(row%4) + kpack       // kpack=0..3，每个代表8个FP16
physical_vector = group*16 + (col XOR (group%16))
```

关闭 XOR 时去掉最后的异或。这一映射每 group 只排列 16 个位置，绝不跨 group，因此是双射：没有丢数据、没有两名生产者写同一个地址，也不需要额外 padding。XOR 还是自己的逆变换。

注意三层地址不能混用：global 地址是普通矩阵坐标；BSM 地址是 swizzled 物理位置；MMA operand 是寄存器中分布到各 lane 的 fragment。

### [06] builtin 搬运与 lane 隐式偏移

`copy_bsm` 使用实际接口：

```cpp
__builtin_mxc_ldg_b128_bsm_predicator(
    dst, src, 0, true, true, false, true, 1, 1, MACA_ICMP_EQ);
```

参数组合取自公开 native kernel 的无条件 128-bit BSM 搬运用法。这里不臆造各布尔位的硬件含义；它们应按目标 SDK 的 builtin 定义解释。`1==1` 使该示例的 predication 条件恒真，尺寸已满足整 tile，不需要边缘补零。

**dst 是 wave 内一致的起点，参考 C500 ABI 由 lane 隐式增加 `lane*16` 字节。** 因而主机推导出的每线程物理位置虽然是 `h*Threads+tid`，传给 builtin 的目的偏移却是 `(h*Threads+wave*64)*16`。再加一次 lane 偏移会造成错位甚至越界。

每个 producer 先由物理位置反推 `row,kpack`，然后从普通 global 地址读取正确的八个 FP16。Swizzle 由地址映射完成，builtin 负责搬运；builtin 不会替你理解矩阵的 M/N/K。

BM=128、256线程时，每矩阵512个向量，需要每线程搬两次，A/B合计每wave每stage四条LDG。这一数量用于等待计数；改变分片深度、线程数或搬运指令后，需要一起修改。

mode 1 则执行标量 global→寄存器→BSM，供功能与性能对比。BSM builtin 可省掉显式中转寄存器并允许异步搬运，但实际吞吐收益需要测量。

### [07] BSM→寄存器

`shared_fragments` 按 [04] 的数学行号、K pack，通过 [05] 的映射找到物理地址，再读取 `U4`。因此 mode 0 与 staged 版本喂给 MMA 的数据相同。

BSM 保存 CTA 的整块 A/B 分片，多个 wave 复用它们：同一 wave 行内复用 A，同一 wave 列内复用 B。这里把“global 合并搬运”与“按 MMA lane 布局消费”分开，BSM 就是两种访问方式之间的中转层。

### bank conflict：本例做了什么，哪些必须实测

不做 XOR 时，固定 i 的 load 中，很多 lane 的 vector 起始 bank 重复。XOR 使 `lane%16` 参与列低位，打散热点。

作为解释模型，假设 32 个 bank、每 bank 4 字节，则一个 16-byte 向量占四个 bank；64 个 lane 的宽向量访问本来就不能在一个 bank 周期内全部完成。未 XOR 时，固定 i 的 vector 起点只有四类，每类16个 lane；XOR 后起点分散到八类，每类8个 lane。它消除了这个模型中的额外集中，**宽度造成的基本分批不应误计成可消除的 conflict**。

这只是说明地址变换的模型，**并未核实当前 C500 SKU 的实际 bank 数量、子组服务规则和指令拆分方式**。不能直接宣称“C500 bank conflict=0”。真正的完成条件是：对 mode 2/3 相同访存指令采样，确认 profiler 的 BSM conflict/replay/吞吐计数改善，且耗时下降。字段名称以安装的 MACA profiler 为准。

### [08] 为什么既要 wait 又要 barrier

异步搬运发出后，BSM 不一定立刻可读。`arrive_gvmcnt(n)` 等待 global memory 操作剩余数量降到阈值；示例采用公开 native kernel 的“保留较新的若干条操作在途”的方式。

`newer_stages` 是当前片后面还有多少个已发出的片；阈值为 `newer_stages*Ops`。其中 Ops 是每 wave 每 stage 的 LDG 指令数，不是字节数、线程数或矩阵数。

builtin 的阈值需要编译期常量，所以用 switch 生成有限的立即数版本。随后调用 block barrier，让不同 wave 的 producer 与 consumer 汇合。仅靠某个 wave 自己等完，不能证明其他 wave 写入的数据已经可供整个 CTA 消费。

此实现依赖参考 SDK 的 LDG 队列、计数器与 barrier 可见性语义；主机模型不模拟硬件完成顺序。换 SDK、增加额外 global load 或改动发出指令数时，必须重新核对等待公式及反汇编。

### [09] store D 与 βC

输入 fragment 和输出 fragment 的 lane layout 不同。对于 `acc.c[i][j][z]`：

```text
m = wave_m_origin + 4*(lane%16) + i
n = wave_n_origin + 16*(lane/16) + 4*z + j
```

64 lane × 4 × 4 × 4 = 4096 个唯一输出，恰好覆盖 64×64，没有原子操作或跨 wave 归约。

先乘 α；β 非零时读取 C 并使用 `fmaf` 加上 βC；最后写 FP32 D。β=0 时允许跳过 C 的读取，因此计时和流量都可能不同。输出当前按标量 store 编写，便于核对坐标；后续可以比较重排后的向量 store，但不能破坏 lane 映射。

### [10] CTA 到逻辑 tile 的坐标

`blockIdx.x/y`选择一个128×128输出区域，起点为`n0=blockIdx.x*128`、`m0=blockIdx.y*128`。归约K的所有逻辑tile都在同一个CTA内累加完成。

每个CTA内`w=tid/64`，`wm=w/2`、`wn=w%2`，对应2×2 wave。每个wave负责64×64区域，四个区域合起来正好是128×128，互不重叠。

`direct_kernel` 显式写了 K 外层256、内层32。staged 版本将这两层展平成 `tiles=K/32`，便于跨逻辑 tile 保持流水线，数学工作量不变。

### [11] 多 stage 循环缓冲区

一个 stage 存放 A[BM,32] 与 B[32,BM]，其字节数为：

```text
StageBytes = 2个矩阵 × BM × 32 × 2字节
BM=128：16384字节
```

普通 staged 版本的循环如下：

1. Prologue 预发 `Stages-1` 个分片；1-stage 时不预发。
2. 第 t 轮再发出最远的 `t+Stages-1` 分片，使未来 global 读取有机会与当前计算重叠。
3. 等待当前 t 分片完成，读取 `slot=t%Stages`。
4. 执行两组 MMA。
5. 全 block barrier，确保任何 wave 都不再读取当前槽位，下一轮才允许复用。
6. 末尾不再发越界分片，并逐步减小 wait 阈值直至0，完成 drain。

例如3-stage：初始发0/1；第0轮发2、消费0；第1轮发3复用slot0、消费1；第2轮发4复用slot1、消费2。保护槽位的 barrier 与数据完成的 wait 解决不同问题，都需要保留。

K 最小为256，即至少8个分片，始终多于本例最多4个 stage，所以 prologue 不会越界。若以后允许 K=32 等小尺寸，必须给 prologue 加边界处理，不能直接复用当前假设。

多 stage 的收益来自隐藏访存延迟；代价包括 BSM 占用增加、resident CTA 减少、屏障和代码调度成本。4-stage并非必胜。

### [12] 寄存器预取与调度候选

mode 7 使用 `current` 和 `next` 两份 fragment：先执行 current 的第一半 MMA，再等待/读取下一片 BSM 数据，接着执行 current 的第二半 MMA。下一片的共享读取与当前片的计算形成可调度的独立工作。

在 current 不再需要 BSM 后，barrier 允许回收它的槽位并发出更远片的 global 读取。下一轮直接使用提前读入的寄存器 fragment。

这一策略让 LDS 延迟有机会被隐藏，但也增加寄存器存活范围，并提前等待下一片。因此它可能更快，也可能因寄存器 spill 或额外等待而更慢。它是供 profiler/benchmark 选择的 pipeline 候选，**尚未在 C500 上完成调优验收**。

### [13] 行主序 B 的转换

`pack_b_kernel` 将 `row[k*N+n]` 写到 `col[n*K+k]`。示例版本以连续写为主，读取端有跨度；它是正确性入口，尚未做共享内存分块转置优化。

权重 B 可以跨多次 GEMM 复用时，可以只转换一次；B 每次改变时，应比较 `pack_plus_gemm_ms`。两种场景不能只拿预处理后的纯 kernel 时间混在一起比较。

### [14] 版本分发与集成位置

`launch` 根据 mode 实例化同一个模板，保证大部分逻辑一致。它是本文件内部的 host 调用入口；要接入项目，可保留类型、device helpers、kernel 和 `launch`，去掉 benchmark 的 `main`。生产接口应补充外部调用参数检查、stream、错误返回和必要的布局转换策略。

benchmark 在启动前检查尺寸、wave64、grid 和每个版本的 threads/BSM 基本限制。编译器报告的寄存器数、spill 和每 kernel 可启动资源仍需要单独核对；设备属性检查无法预先证明这些资源一定足够。

### [15] 独立正确性参考

参考 kernel 用普通 SIMT FP64 乘加，直接按数学定义计算，不使用 MMA、不读取 swizzled BSM，避免两个实现共享同一布局错误。

输出元素数不超过262144时全量检查；更大输出按行主序索引分层抽4096点。大尺寸抽样不能代替全量验收，因此脚本先执行多组小尺寸全量测试。随机输入的 FP16 位模式来自主机转换，参考端读取同一输入。

判断条件为：

```text
abs(D - ref) <= 2e-3 + 2e-3*abs(ref)
```

这是针对本例输入规模的工程阈值，并非任意 K、任意动态范围的浮点误差保证。近零结果、强抵消、极大输入或更长 K 需要单独制定精度要求。FP32 MMA 的累加顺序与 FP64 参考不同，不能要求 bitwise equality。

每次先把 D 填成 NaN 位模式，防止漏写位置依赖旧内存偶然过关。脚本还包含全1、零A、类单位A、α=0、β=0/非零和矩形尺寸。

### [16] 时间测量与输出

每个版本先校验，通过后 warmup 5 次，进行7轮 GPU event 测量，每轮运行指定 repeats 次，报告每次调用平均时间的轮间中位数。C 一直保持原始值，输出独立写 D。

GPU event区间不含H2D/D2H、初始化和参考计算；`--row-b`另测“转换+GEMM”。重复读取同一输入可能形成热缓存，结果应标注为重复运行吞吐。很小的128×128×256问题也可能受到launch/主机供给间隙影响，可在实测时增加repeats，并用SDK支持的graph或profiler核对设备时间。

CSV 字段：`mode,name,check,max_abs,max_scaled,kernel_ms,TFLOPS,peak_pct,pack_plus_gemm_ms`。设备信息和说明行也会写到标准输出，若后处理为CSV应筛选以数字 mode 开头且列数完整的行。

## 4. 最终衡量指标：性能由耗时决定，利用率帮助解释

### 首先是正确性

只有满足形状、类型、α/β语义和精度阈值的实现才进入性能比较。正确性失败的高 TFLOPS 没有意义。

### 主指标一：延迟 / 吞吐

固定问题尺寸、相同输入精度和输出语义，直接比较 GPU 延迟 `t`。更小的 t 更好。

矩阵乘加的常用主项计数是 `2MNK` FLOPs，FMA按2个浮点操作计算：

```text
TFLOPS = 2*M*N*K / (t_ms * 10^9)
```

默认不把epilogue的O(MN)运算混入主项，这与常见GEMM报告口径一致。128×128×256的工作量为8388608 FLOPs。假设测得0.01ms，则为0.8388608 TFLOPS；**这里只是公式示例，不是本程序实测结果**。

### 主指标二：相对库效率与峰值算力利用率

有 mcBLAS 数据时，优先给出：

```text
相对库效率 = 本kernel TFLOPS / 同形状mcBLAS TFLOPS
           = mcBLAS时间 / 本kernel时间
峰值算力利用率 = 本kernel TFLOPS / 本卡dense FP16→FP32理论峰值
```

二者不能混同。mcBLAS也可能没有达到理论峰值。当前交付的 benchmark **未集成或实测 mcBLAS**，库效率需另行采集；不能凭空填一个“90%库效率”。

mcBLAS按列主序时，本例行主序D可以转化成 `Dᵀ=BᵀAᵀ`：调用 GEMM `(OP_T, OP_N, N, M, K)`，第一个 operand 为 B、ld=K，第二个为 A、ld=K，输出 ld=N，输入16F、输出32F、计算32F。α/β与本例一致；β非零时需保证每次测量的初始C一致。转换、重置输出的时间是否计入，也必须统一。

峰值分母必须对应**具体C500型号、频率、dense输入类型、FP32累加口径**；不能拿稀疏2:4峰值、INT8峰值或其他GPU的峰值做分母。`--peak=` 仅接受你确认的分母，未设置时输出NA；本例不硬编码一个未核实的峰值。

### Occupancy、GPU utilization 与 MMA 活跃度都是诊断指标

| 指标 | 能解释什么 | 为什么不能单独定胜负 |
|---|---|---|
| Occupancy | 活跃wave占支持上限的比例 | 更高occupancy可能换来更小tile、较差复用；更低也可能已充分隐藏延迟 |
| 管理工具GPU利用率 | 采样窗口里GPU有工作的时间比例等 | 忙于搬运、等待或低效kernel也可能显示很高，且语义依工具而异 |
| MMA pipe active/issue | 矩阵执行管线忙碌程度 | 更接近算力瓶颈，但仍要结合频率和实际工作量 |
| BSM冲突/重放 | shared布局是否导致额外服务 | 计数减少后仍需看总延迟是否改善 |
| global吞吐/缓存命中 | global搬运压力、cache复用 | 高带宽也可能源于重复搬运，并不必然高效 |
| wait/barrier stall | 流水线是否充分重叠 | 要区分global等待、shared等待与不同wave进度不齐 |
| 寄存器、spill、local流量 | tile和prefetch是否超出资源 | 提高stage或预取深度可能引发反向收益 |

### Roofline 作方向判断

从DRAM至少读一次A/B、读C并写D的理想流量：

```text
bytes_min = 2*M*K + 2*K*N + 4*M*N + (beta!=0 ? 4*M*N : 0)
AI_ideal  = 2*M*N*K / bytes_min
性能上界 ≲ min(dense算力峰值, 实测可持续DRAM带宽*AI)
```

这只是理想下界流量。CTA之间重复载入、cache容量、B转换都会改变实际DRAM字节数；应以profiler实际流量算AI。对于孤立128×128×256问题，β非零时理想AI为32 FLOP/byte，β=0时约42.67；更大问题的AI会不同。

## 5. 如何真正完成性能优化验收

1. 先在C500运行全部小尺寸正确性检查。若 mode0失败，先核对MMA/lane/output ABI；若mode1通过而mode2失败，重点核对BSM builtin地址与等待语义。
2. 固定电源/频率配置和空闲卡，记录卡型号、SDK、驱动、编译命令，采集重复测量中位数和波动。
3. mode1→2比较builtin；mode2→3比较XOR。用ISA确认实际向量load、LDS和MMA，不仅看C++源码。
4. mode3→4→5→6比较stage；结合BSM使用、resident block、stall和spill决定stage数。
5. mode5→7比较寄存器预取。若spill增加或wait提前造成停顿，则保留mode5或调整发出/等待位置。
6. 在多个M/N/K形状上选型，而非仅128×128×256。最小问题只有一个CTA，无法代表大矩阵的整卡吞吐；不要为了“凑满SM”启动重复计算且覆盖同一D的block。
7. 同口径测mcBLAS，并在实际接口下比较转换+计算。在固定128×128×256 tile约束下，还可继续调SliceK、epilogue向量化和grid swizzle。

推荐实验表：

| shape | mode | correctness | median ms | TFLOPS | /mcBLAS | peak% | regs/thread | spill | BSM conflict | 说明 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128×128×256 | 待测 | 待测 | — | — | — | — | — | — | — | 单CTA延迟 |
| 1024³ | 待测 | 待测 | — | — | — | — | — | — | — | 过渡尺寸 |
| 4096³ | 待测 | 待测 | — | — | — | — | — | — | — | 大尺寸吞吐 |
| 2048×4096×1024 | 待测 | 待测 | — | — | — | — | — | — | — | 矩形 |

## 6. 此环境已执行的验证及边界

运行`node validate_layout.mjs`，检查BM=128、XOR开关两种情况的搬运双射、参考native MMA数学布局、每个输出恰好写一次；以及K=256/512/768/1024、stage1–4的普通/寄存器预取环形缓冲区不提前覆盖、prologue/drain不越界。结果通过。

该脚本是数学模型，不执行 `.cu`，也无法验证实际指令、bank结构、内存可见性、寄存器分配或数值误差。当前没有真机计时数据，没有宣称某一mode最快。最终验收仍需运行`build.sh`、`run_checks.sh`与性能实验。

## 7. 依据与来源

接口依据固定到公开工程提交 `bb09c70513d08903b8d9cee2774c4e0062e6c4f5`，便于复核：

- [native MMA类型与BSM宏](https://github.com/QingCheng-AI/muxi_native_layout_kernels/blob/bb09c70513d08903b8d9cee2774c4e0062e6c4f5/csrc/muxi_hgemm_utils.cuh)
- [C500 native GEMM：wave64、异步搬运、等待、输出映射](https://github.com/QingCheng-AI/muxi_native_layout_kernels/blob/bb09c70513d08903b8d9cee2774c4e0062e6c4f5/csrc/muxi_hgemm.cuh)
- [编译参数](https://github.com/QingCheng-AI/muxi_native_layout_kernels/blob/bb09c70513d08903b8d9cee2774c4e0062e6c4f5/setup.py)
- [mcBLAS GEMM接口用法](https://github.com/QingCheng-AI/muxi_native_layout_kernels/blob/bb09c70513d08903b8d9cee2774c4e0062e6c4f5/csrc/gemm_ex.cu)
- [沐曦运行时API编程指南](https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html)
- [沐曦官方C500产品页](https://www.metax-tech.com/en/goods/prod.html?cid=107&id=21)

XOR布局和本例流水线由上述接口契约推导，属于本实现的设计，不能理解为厂商已经验证或推荐的最优kernel。
