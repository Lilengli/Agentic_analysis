"""Run repeated causal attention for external msprof system sampling.

This program deliberately does not start torch_npu.profiler: msprof owns
collection, and the elapsed time reported here is not a benchmark result.
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impl', choices=('triton', 'sdpa'), required=True)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='calls between device synchronizations')
    args = parser.parse_args()
    if not (0 < args.seconds <= 3600) or args.warmup < 1 or args.batch_size < 1:
        parser.error('seconds must be in (0, 3600]; warmup and batch-size must be positive')

    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend
    import torch.nn.functional as F
    from flash_attention_ascend import kernel_function

    if not torch.npu.is_available():
        raise RuntimeError('No NPU is available to this Python process')

    torch.manual_seed(1234)
    shape = (1, 2, 128, 128)
    q, k, v = (torch.randn(shape, dtype=torch.float16, device='npu:0')
               for _ in range(3))

    def run():
        if args.impl == 'triton':
            return kernel_function(q, k, v)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = run()
        torch.npu.synchronize()
        start = time.monotonic()
        deadline = start + args.seconds
        calls = 0
        print(f'sampling workload start: impl={args.impl} shape={shape} '
              f'dtype=float16 causal=True duration={args.seconds}s', flush=True)
        while time.monotonic() < deadline:
            for _ in range(args.batch_size):
                output = run()
            torch.npu.synchronize()
            calls += args.batch_size
        elapsed = time.monotonic() - start
        print(f'sampling workload end: impl={args.impl} calls={calls} '
              f'elapsed={elapsed:.3f}s (not a benchmark)', flush=True)


if __name__ == '__main__':
    main()
