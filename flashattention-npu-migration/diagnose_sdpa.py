"""Independently check long-context SDPA's outputs and causal visibility."""
import argparse
import math


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq', type=int, default=16384)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--atol', type=float, default=0.002)
    p.add_argument('--rtol', type=float, default=0.01)
    a = p.parse_args()
    if a.seq < 2 or a.rows < 2 or a.atol < 0 or a.rtol < 0:
        p.error('seq and rows must be at least 2; tolerances must be non-negative')
    import torch
    import torch_npu  # noqa: F401
    import torch.nn.functional as F
    if not torch.npu.is_available():
        raise RuntimeError('NPU is unavailable')
    torch.manual_seed(a.seed)
    shape = (1, 2, a.seq, 128)
    q_cpu, k_cpu, v_cpu = [torch.randn(shape, dtype=torch.float16) for _ in range(3)]
    q, k, v = (x.to('npu:0') for x in (q_cpu, k_cpu, v_cpu))
    # Include early, middle and late positions with their entire valid history.
    rows = torch.linspace(0, a.seq - 1, min(a.seq, a.rows)).round().long().unique()
    scores = (q_cpu.float().index_select(2, rows) @ k_cpu.float().transpose(-1, -2)) / math.sqrt(128)
    mask = torch.arange(a.seq)[None, :] > rows[:, None]
    reference = torch.softmax(scores.masked_fill(mask, float('-inf')), -1) @ v_cpu.float()
    with torch.inference_mode():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        torch.npu.synchronize()
        selected = out.cpu().float().index_select(2, rows)
        diff = selected - reference
        rmse = diff.square().mean().sqrt().item()
        ref_rms = reference.square().mean().sqrt().item()
        passed = torch.allclose(selected, reference, atol=a.atol, rtol=a.rtol)
        zero_passes = torch.allclose(torch.zeros_like(reference), reference, atol=a.atol, rtol=a.rtol)
        print(f'shape={shape} checked_rows={len(rows)}/{a.seq} seed={a.seed}')
        print(f'atol={a.atol} rtol={a.rtol} allclose={passed}')
        print(f'max_abs={diff.abs().max().item():.8g} mean_abs={diff.abs().mean().item():.8g} '
              f'rmse={rmse:.8g} reference_rms={ref_rms:.8g} normalized_rmse={rmse/max(ref_rms, 1e-30):.8g}')
        print(f'all_zero_output_passes_reference_check={zero_passes}')

        # Changing future V tokens must leave earlier outputs unchanged, while
        # causing a visible change in at least some later outputs.
        boundary = a.seq // 2
        v_changed = v.clone()
        v_changed[:, :, boundary:, :] += 2.0
        changed = F.scaled_dot_product_attention(q, k, v_changed, is_causal=True, dropout_p=0.0)
        torch.npu.synchronize()
        early = (out[:, :, :boundary, :].float() - changed[:, :, :boundary, :].float()).abs().max().item()
        late = (out[:, :, -1, :].float() - changed[:, :, -1, :].float()).abs().mean().item()
        print(f'future_V_perturbation: early_max_abs_change={early:.8g}, '
              f'last_token_mean_abs_change={late:.8g}')
        causal_pass = early <= a.atol and late > 0.1
        print(f'causal_perturbation_check={causal_pass}')
        if not passed or not causal_pass:
            raise SystemExit('Diagnostic failed; inspect results before comparing performance')


if __name__ == '__main__':
    main()
