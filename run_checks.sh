#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH="${MACA_PATH:-/opt/maca}/lib:${MACA_PATH:-/opt/maca}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
# Small cases are checked fully, including rectangular shapes and K-tile tails.
# -1 runs all eight variants, so a wrong layout cannot hide behind a fast mode.
./c500_gemm 128 128 256 -1 2 1 0
./c500_gemm 384 128 768 -1 2 0.75 -0.25
./c500_gemm 128 640 512 -1 2 -0.5 1 --row-b
./c500_gemm 512 512 512 -1 2 1 1
./c500_gemm 128 128 256 -1 2 1 1 --pattern=ones
./c500_gemm 128 128 256 -1 2 1 0 --pattern=identity
./c500_gemm 128 128 256 -1 2 1 1 --pattern=zero
./c500_gemm 128 128 256 -1 2 0 1
# These must be rejected before allocating/launching.
if ./c500_gemm 129 128 256 0 1 1 0; then exit 1; fi
if ./c500_gemm 128 128 128 0 1 1 0; then exit 1; fi
if ./c500_gemm 128 128 256 8 1 1 0; then exit 1; fi
echo "C500 functional checks passed (inspect UNSUPPORTED_DEVICE_RESOURCES rows)."
