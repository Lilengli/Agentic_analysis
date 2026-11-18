# CATLASS hardware primitive headers

`catlass_headers.tar.gz` contains the unmodified `include/` tree and `LICENSE`
from https://gitcode.com/cann/catlass at revision
`769cd40a8716b28650b6bebb08db4834eea4462f`, obtained from the CATLASS submodule
of the local flash-attention-npu checkout. Copyright notices remain in every
upstream file. The bundled software is governed by the accompanying CANN Open
Software License Agreement Version 2.0, not by any different parent license.

SHA256: `7f99e4ec0f09f6ef2c5278218faa44785aec7971273e1237778513dc7e137767`.
Only its tile copy/layout/Mmad primitives are used. This implementation does
not call or copy the existing CATLASS/flash-attention-npu attention operator.
The bundle avoids a new network dependency on the target machine.
