#!/bin/bash
# asm2.sh <src.cu> <arch> <outprefix> [extra mxcc flags] -> /tmp/<outprefix>.s
SRC=$1; ARCH=$2; OUT=$3; shift 3
M=/opt/maca/mxgpu_llvm/bin/mxcc
BIN=/opt/maca/mxgpu_llvm/bin
TGT=maca-mxc-metax-macahca--$ARCH-bc
$M -x maca -O3 -std=c++17 --offload-arch=$ARCH --maca-path=/opt/maca "$@" -fatbc "$SRC" -o /tmp/$OUT.bc 2>/tmp/$OUT.err1
$BIN/clang-offload-bundler --type=bc --unbundle --input=/tmp/$OUT.bc --targets=$TGT --output=/tmp/$OUT.d.bc 2>/tmp/$OUT.err2
$BIN/llvm-dis /tmp/$OUT.d.bc -o /tmp/$OUT.ll 2>/tmp/$OUT.err3
$M -cc1 -triple mxc-metax-macahca -target-cpu $ARCH -S -O3 -x ir /tmp/$OUT.ll -o /tmp/$OUT.s 2>/tmp/$OUT.err4
if [ ! -s /tmp/$OUT.s ]; then
  echo "FAILED $OUT"; for e in 1 2 3 4; do echo "-- err$e:"; head -3 /tmp/$OUT.err$e; done; exit 1
fi
echo "$OUT.s: $(wc -l < /tmp/$OUT.s) 行"
