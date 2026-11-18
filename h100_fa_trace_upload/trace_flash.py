import json
import os
from pathlib import Path

import torch
import flash_attn
from flash_attn import flash_attn_func

root = Path(os.environ["TRACE_TASK_DIR"])
shape = (1, 16384, 2, 128)  # B, S, H, D

torch.manual_seed(1234)
q, k, v = [
    torch.randn(shape, device="cuda", dtype=torch.float16)
    for _ in range(3)
]

torch.cuda.synchronize()

with torch.inference_mode():
    out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

torch.cuda.synchronize()

def info(name, tensor):
    start = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "start": hex(start),
        "end": hex(start + size),
        "bytes": size,
    }

metadata = {
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flash_attn_version": flash_attn.__version__,
    "flash_attn_path": flash_attn.__file__,
    "causal": True,
    "dropout_p": 0.0,
    "seed": 1234,
    "tensors": [info("Q", q), info("K", k), info("V", v), info("O", out)],
}

(root / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("Output shape:", tuple(out.shape))
print("Metadata:", root / "metadata.json")
