"""Experimental Ascend autotuning for the migrated causal attention kernel.

Calibrates on the first input of each exact workload in this process. Candidate
correctness is sampled for long sequences; this is not production validation.
"""
import math
import time

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton

from flash_attention_ascend import _causal_attention_kernel, kernel_function

_CANDIDATES = ((32, 32), (64, 32), (64, 64), (128, 64))
_CACHE = {}


def _workload_key(q, k, v):
    return (str(q.device), str(q.dtype), tuple(q.shape), tuple(q.stride()),
            tuple(k.stride()), tuple(v.stride()))


def _reference(q, k, v, max_rows):
    n = q.shape[2]
    count = n if n <= 512 else min(n, max_rows)
    rows = torch.linspace(0, n - 1, count).round().long().unique()
    q_cpu, k_cpu, v_cpu = (x.detach().cpu().float() for x in (q, k, v))
    logits = q_cpu.index_select(2, rows) @ k_cpu.transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.arange(n)[None, :] > rows[:, None]
    return rows, torch.softmax(logits.masked_fill(mask, float('-inf')), -1) @ v_cpu


def _make_autotuner(valid):
    configs = [triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}) for bm, bn in valid]
    # Use the Ascend backend's Triton autotuner. A separate autotuner is cached
    # for each full workload/device/stride tuple, including batch and heads.
    return triton.autotune(configs=configs, key=['N', 'D', 'H'])(_causal_attention_kernel)


def _launch(tuner, q, k, v):
    batch, heads, seq, dim = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(seq, meta['BLOCK_M']) * batch * heads,)
    tuner[grid](q, k, v, out, heads, seq, dim, *q.stride(), *k.stride(), *v.stride(),
                BLOCK_D=max(16, triton.next_power_of_2(dim)))
    return out


def autotuned_attention(q, k, v, *, reference_rows=64, atol=0.002, rtol=0.01):
    """Run causal FP16 attention, calibrating and autotuning on the first call.

    First-call output must not be used for steady-state timing. Fixed candidate
    configs are checked against a CPU FP32 reference for this input, then Triton
    benchmarks the passing configs. Not thread-safe; cache is in-process only.
    """
    for x in (q, k, v):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.device.type != 'npu' or x.dtype != torch.float16:
            raise ValueError('q, k, v must be 4-D FP16 NPU tensors')
    if q.shape != k.shape or q.shape != v.shape or q.device != k.device or q.device != v.device:
        raise ValueError('q, k, v must have the same shape and device')
    if min(q.shape) < 1 or q.shape[-1] > 256 or reference_rows < 1 or atol < 0 or rtol < 0:
        raise ValueError('shape/reference_rows/tolerances outside supported range')
    key = (_workload_key(q, k, v), int(reference_rows), float(atol), float(rtol))
    if key not in _CACHE:
        started = time.monotonic()
        rows, reference = _reference(q, k, v, reference_rows)
        candidates = []
        valid = []
        for bm, bn in _CANDIDATES:
            item = {'block_m': bm, 'block_n': bn}
            try:
                actual = kernel_function(q, k, v, block_m=bm, block_n=bn)
                torch.npu.synchronize()
                selected = actual.detach().cpu().float().index_select(2, rows)
                item['max_abs_error'] = (selected - reference).abs().max().item()
                item['correct'] = bool(torch.allclose(selected, reference, atol=atol, rtol=rtol))
                if item['correct']:
                    valid.append((bm, bn))
            except Exception as exc:
                item.update(correct=False, error=repr(exc))
            candidates.append(item)
            print(f'autotune candidate {bm}x{bn}: {item}', flush=True)
        if not valid:
            raise RuntimeError(f'No autotune candidates passed correctness: {candidates}')
        tuner = _make_autotuner(valid)
        # First call benchmarks candidates and executes the selected kernel.
        out = _launch(tuner, q, k, v)
        torch.npu.synchronize()
        selected_output = out.detach().cpu().float().index_select(2, rows)
        if not torch.allclose(selected_output, reference, atol=atol, rtol=rtol):
            raise RuntimeError('Autotuned output failed CPU reference check after selection')
        chosen = getattr(tuner, 'best_config', None)
        chosen_kwargs = getattr(chosen, 'kwargs', None)
        if chosen_kwargs is None and len(valid) == 1:
            chosen_kwargs = {'BLOCK_M': valid[0][0], 'BLOCK_N': valid[0][1]}
        report = {'shape': list(q.shape), 'device': str(q.device), 'dtype': str(q.dtype),
                  'reference_rows_checked': rows.tolist(), 'atol': atol, 'rtol': rtol,
                  'candidates': candidates, 'selected': dict(chosen_kwargs) if chosen_kwargs else None,
                  'calibration_seconds': time.monotonic() - started}
        _CACHE[key] = (tuner, report)
        print(f'autotune selected: {report["selected"]}; calibration={report["calibration_seconds"]:.3f}s', flush=True)
        return out
    return _launch(_CACHE[key][0], q, k, v)


def tuning_report(q, k, v, *, reference_rows=64, atol=0.002, rtol=0.01):
    """Return calibration metadata for this exact workload, or None."""
    key = (_workload_key(q, k, v), int(reference_rows), float(atol), float(rtol))
    entry = _CACHE.get(key)
    return entry[1] if entry else None
