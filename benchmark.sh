#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH="${MACA_PATH:-/opt/maca}/lib:${MACA_PATH:-/opt/maca}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
# Optional: PEAK_TFLOPS must be the DENSE FP16->FP32 peak of this card/config.
extra=()
if [[ -n "${PEAK_TFLOPS:-}" ]]; then extra+=("--peak=${PEAK_TFLOPS}"); fi
for shape in "128 128 256" "1024 1024 1024" "4096 4096 4096" "2048 4096 1024"; do
  read -r m n k <<< "$shape"
  ./c500_gemm "$m" "$n" "$k" -1 30 1 0 "${extra[@]}"
done
# Measure conversion overhead separately for a row-major B interface.
./c500_gemm 4096 4096 4096 -1 30 1 0 --row-b "${extra[@]}"
