# =============================================================================
# 沐曦 C500 GEMM 构建脚本
#
#   make                 # 默认:HAS_MMA=0,只编 v0 基线,用来先验证工具链通
#   make HAS_MMA=1       # 开 MMA kernel(需先核对 builtin 名字,见 mx_mma.h)
#   make HAS_MMA=1 ASYNC=1 BT=1   # 全开:异步 bsm 拷贝 + B 预转置布局
#   make probe           # 只跑片段布局自检
#   make bench           # 跑 4096^3 性能对比
# =============================================================================

# 沐曦编译器。cucc 是 CUDA 兼容前端(认 cudaMalloc/<<<>>>);
# 若你的 SDK 只提供 mxcc,用: CXX = mxcc -x maca  并加 -DMX_NATIVE_MACA
CXX      ?= cucc
ARCH     ?= mxgpu_c500        # 真实名字用 `cucc --help | grep -i arch` 或 mx-smi 确认

HAS_MMA  ?= 0
ASYNC    ?= 0
BT       ?= 0
LAYOUT   ?= 0
WAVE     ?= 64
# stage 数不是编译宏:v3/v4 两个实例在 gemm_c500.cu 的 main 里直接模板实例化,
# 想试 2 stage 就在那里加一行 gemm_mma<2,1,1>

DEFS  = -DMX_HAS_MMA=$(HAS_MMA) -DMX_HAS_BSM_ASYNC=$(ASYNC) \
        -DMX_B_COLMAJOR=$(BT) -DMX_FRAG_LAYOUT=$(LAYOUT) -DMX_WAVE=$(WAVE)

# -O3 必须开,否则 #pragma unroll 不展开、累加器会落到栈上
# --save-temps 保留汇编,用来确认 MMA 条数、LDS 指令宽度(ds_read_b64 而非 b16)
FLAGS = -O3 -std=c++14 --offload-arch=$(ARCH) $(DEFS)

TARGET = gemm_c500

all: $(TARGET)

$(TARGET): gemm_c500.cu mx_mma.h
	$(CXX) $(FLAGS) -o $@ gemm_c500.cu

# 看寄存器用量和 spill —— 这是本 kernel 最需要盯的编译产物
res: gemm_c500.cu mx_mma.h
	$(CXX) $(FLAGS) -Rpass-analysis=kernel-resource-usage -c gemm_c500.cu -o /dev/null

# 导出汇编,检查 MMA 指令真的发出来了、LDS 读是 64 位宽
asm: gemm_c500.cu mx_mma.h
	$(CXX) $(FLAGS) -S -o $(TARGET).s gemm_c500.cu
	@grep -c -i "mma" $(TARGET).s | xargs echo "MMA 指令条数:"
	@grep -c -i "ds_read" $(TARGET).s | xargs echo "LDS 读指令条数:"

probe: $(TARGET)
	./$(TARGET) --probe -m 128 -n 128 -k 256 --iters 1

# 小规模 + CPU 参考对数
check: $(TARGET)
	./$(TARGET) -m 512 -n 512 -k 512 --iters 5

# 性能:PEAK / BWPEAK 请填 C500 规格书上的 fp16 峰值算力和 HBM 峰值带宽
PEAK   ?= 0
BWPEAK ?= 0
bench: $(TARGET)
	./$(TARGET) -m 4096 -n 4096 -k 4096 --iters 50 --peak $(PEAK) --bw $(BWPEAK)

clean:
	rm -f $(TARGET) $(TARGET).s *.o

.PHONY: all res asm probe check bench clean
