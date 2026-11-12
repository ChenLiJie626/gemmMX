#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
MACA_PATH="${MACA_PATH:-/opt/maca}"
MXCC="${MXCC:-${MACA_PATH}/mxgpu_llvm/bin/mxcc}"
if [[ ! -x "$MXCC" ]]; then
  MXCC="$(command -v mxcc || true)"
fi
if [[ -z "$MXCC" || ! -x "$MXCC" ]]; then
  echo "mxcc not found: set MACA_PATH or MXCC to your C500 SDK compiler" >&2
  exit 2
fi
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
# Flags match the referenced native MACA kernel project's BSM addressing setup.
# Extra command-line arguments allow SDK-specific target/resource-report flags.
"$MXCC" -x maca -std=c++17 -O3 \
  -mllvm -metaxgpu-disable-bsm-offset=0 \
  -mllvm -metaxgpu-force-global-saddr=1 \
  -I"${MACA_PATH}/include" -L"${MACA_PATH}/lib" \
  c500_gemm.cu -o c500_gemm "$@"
# mxcc normally links the native runtime automatically. If your SDK differs,
# use its documented runtime linker flags as extra arguments above.
