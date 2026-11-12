# Source acknowledgments

This independently structured tutorial uses the C500 native intrinsic calling
conventions, wave64 fragment/epilogue mapping, and compiler flag examples published
in QingCheng-AI/muxi_native_layout_kernels, commit
bb09c70513d08903b8d9cee2774c4e0062e6c4f5.

Upstream: https://github.com/QingCheng-AI/muxi_native_layout_kernels

The upstream source states that its development was kindly supported by MetaX,
and is licensed under the Apache License, Version 2.0. A copy of that license is
included in LICENSE-APACHE-2.0.txt. No separate upstream NOTICE file was listed
in the inspected tree.

Changes for this tutorial: new 32-wide K slices, layout permutation, scalar and
BSM comparison modes, 128x128x256 logical tiles, stage ring, register-prefetch
candidate, row-major B conversion, verification and timing harness, and Chinese
explanations. This is not an upstream or MetaX-endorsed release.
