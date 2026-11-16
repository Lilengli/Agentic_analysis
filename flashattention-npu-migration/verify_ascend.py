"""Check a first Triton Ascend attention candidate against an FP32 CPU reference."""
import math
import traceback

import torch
import torch_npu  # noqa: F401

from flash_attention_ascend import kernel_function


def reference(q, k, v):
    d = q.shape[-1]
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(d)
    n = q.shape[-2]
    mask = torch.ones((n, n), dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1) @ v.float()


def main():
    if not torch.npu.is_available():
        raise RuntimeError('NPU is not available in this Python environment')
    torch.manual_seed(1234)
    failed = False
    for shape in [(1, 1, 32, 64), (1, 2, 65, 64), (1, 2, 128, 128)]:
        try:
            inputs = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
            expected = reference(*inputs)
            actual = kernel_function(*(x.to('npu:0') for x in inputs))
            torch.npu.synchronize()
            actual = actual.cpu().float()
            error = (actual - expected).abs()
            passed = torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
            print(f'{shape}: {"PASS" if passed else "FAIL"}; '
                  f'max_abs_error={error.max().item():.6g}', flush=True)
            failed |= not passed
        except Exception:
            failed = True
            print(f'{shape}: ERROR', flush=True)
            traceback.print_exc()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
