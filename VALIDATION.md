# Claude implementation snapshot

- Imported on 2026-11-12 from the user-provided `D:\projects\MuXi\gemm` directory.
- `gemm_c500.cu`, `mx_mma.h`, `Makefile`, and `README.md` were copied without implementation edits; source/copy SHA-256 hashes matched before committing. Git normalizes text line endings to LF.
- Target tile: N×M×K = 128×128×256; four wave64 groups, arranged 2×2.
- No MACA compiler or C500 execution environment was detected in this session. No GPU compilation, correctness result, profiler measurement, or performance result is claimed.
- The original README documents unverified builtin names, signatures, fragment layout, bank model, and target architecture name.
- Source review found that `MX_BSM_COPY8`, `MX_BSM_COMMIT`, and `MX_BSM_WAIT` are not called by the GEMM kernel. The current `ASYNC=1` flag therefore does not connect an asynchronous BSM path.
- The numerical comparator does not reject NaN explicitly, and ordinary correctness failures do not change the final process exit code. These behaviors remain in this snapshot and must be addressed before treating its output as acceptance evidence.

This branch preserves the submitted implementation for comparison; the Codex implementation is on the independent `codex` branch.
