#!/bin/bash
# asm.sh <src.cu> <outprefix> [extra mxcc flags]  -> /tmp/<outprefix>.s
SRC=$1; OUT=$2; shift 2
M=/opt/maca/mxgpu_llvm/bin/mxcc
BIN=/opt/maca/mxgpu_llvm/bin
TGT=maca-mxc-metax-macahca--xcore1000-bc
$M -x maca -O3 -std=c++17 -offload-arch native --maca-path=/opt/maca "$@" -fatbc "$SRC" -o /tmp/$OUT.bc 2>/dev/null
$BIN/clang-offload-bundler --type=bc --unbundle --input=/tmp/$OUT.bc --targets=$TGT --output=/tmp/$OUT.d.bc 2>/dev/null
$BIN/llvm-dis /tmp/$OUT.d.bc -o /tmp/$OUT.ll 2>/dev/null
$M -cc1 -triple mxc-metax-macahca -target-cpu xcore1000 -S -O3 -x ir /tmp/$OUT.ll -o /tmp/$OUT.s 2>/dev/null
echo "$OUT.s: $(wc -l < /tmp/$OUT.s) 行"
