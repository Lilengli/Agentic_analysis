"""Check and benchmark DeepSeek-V3/R1 MoE on one Ascend NPU using official CANN ops.

Uses built-in torch_npu operators (no custom C++ extension compilation required):
- Optional gating: torch_npu.npu_moe_gating_top_k
- Routing/permute: torch_npu.npu_moe_init_routing_v2
- Expert FFN:      torch_npu.npu_grouped_matmul + torch_npu.npu_swiglu + torch_npu.npu_grouped_matmul
- Unpermute/merge: torch_npu.npu_moe_token_unpermute (or npu_moe_finalize_routing)
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time


def cpu_gating_topk(torch, logits, bias, topk, n_group, k_group, routed_scaling_factor):
    """DeepSeek-V3 grouped sigmoid top-k routing reference in FP32."""
    scores = torch.sigmoid(logits.float())
    scores_for_choice = scores + bias.float().unsqueeze(0)
    num_tokens, num_experts = scores.shape
    group_size = num_experts // n_group
    group_scores = (
        scores_for_choice.view(num_tokens, n_group, group_size)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = group_scores.topk(k_group, dim=-1, sorted=False)[1]
    group_mask = scores.new_zeros(num_tokens, n_group)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, group_size)
        .reshape(num_tokens, num_experts)
    )
    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))
    topk_ids = masked_scores.topk(topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def build_inputs(
    torch,
    device,
    num_tokens,
    num_experts,
    topk,
    hidden_size,
    intermediate_size,
    dtype,
    seed,
    include_gating,
    include_shared,
    n_group,
    k_group,
    routed_scaling_factor,
):
    import torch.nn.functional as F

    rng = torch.Generator().manual_seed(seed)
    x_cpu = torch.randn(num_tokens, hidden_size, generator=rng, dtype=dtype)
    x_cpu.mul_(1.0 / math.sqrt(hidden_size))

    if include_gating:
        logits_cpu = torch.randn(num_tokens, num_experts, generator=rng, dtype=dtype)
        bias_cpu = torch.randn(num_experts, generator=rng, dtype=dtype)
        bias_cpu.mul_(0.1)
        topk_weights_f32, topk_ids_i64 = cpu_gating_topk(
            torch, logits_cpu, bias_cpu, topk, n_group, k_group, routed_scaling_factor
        )
        topk_ids_cpu = topk_ids_i64.to(torch.int32)
        topk_weights_cpu = topk_weights_f32.to(dtype)
    else:
        logits_cpu = None
        bias_cpu = None
        selected = [torch.randperm(num_experts, generator=rng)[:topk] for _ in range(num_tokens)]
        topk_ids_cpu = torch.stack(selected, dim=0).to(torch.int32)
        raw_weights = torch.rand(num_tokens, topk, generator=rng, dtype=torch.float32) + 0.1
        topk_weights_cpu = (raw_weights / raw_weights.sum(dim=-1, keepdim=True) * routed_scaling_factor).to(dtype)

    if include_shared:
        shared_w13_cpu = torch.randn(hidden_size, 2 * intermediate_size, generator=rng, dtype=dtype)
        shared_w13_cpu.mul_(1.0 / math.sqrt(hidden_size))
        shared_w2_cpu = torch.randn(intermediate_size, hidden_size, generator=rng, dtype=dtype)
        shared_w2_cpu.mul_(1.0 / math.sqrt(intermediate_size))
    else:
        shared_w13_cpu = None
        shared_w2_cpu = None

    # Allocate large expert weights directly on device to avoid huge CPU RNG & RAM overhead.
    if device.type == 'npu':
        torch.npu.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    w13 = torch.randn(num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype)
    w13.mul_(1.0 / math.sqrt(hidden_size))
    w2 = torch.randn(num_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
    w2.mul_(1.0 / math.sqrt(intermediate_size))

    # CPU FP32 reference: only copy weights of experts actually selected by topk_ids.
    x_f32 = x_cpu.float()
    tw_f32 = topk_weights_cpu.float()
    ref = torch.zeros_like(x_f32)
    expert_to_tokens = {}
    for t in range(num_tokens):
        for k_idx in range(topk):
            e = int(topk_ids_cpu[t, k_idx].item())
            expert_to_tokens.setdefault(e, []).append((t, k_idx))
    for e, pairs in expert_to_tokens.items():
        w13_e = w13[e].cpu().float()
        w2_e = w2[e].cpu().float()
        for t, k_idx in pairs:
            gate_up = x_f32[t] @ w13_e
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            ref[t] += tw_f32[t, k_idx] * (act @ w2_e)

    if include_shared:
        s_gate_up = x_f32 @ shared_w13_cpu.float()
        s_gate, s_up = s_gate_up.chunk(2, dim=-1)
        ref = ref + (F.silu(s_gate) * s_up) @ shared_w2_cpu.float()

    x = x_cpu.to(device)
    topk_ids = topk_ids_cpu.to(device)
    topk_weights = topk_weights_cpu.to(device)
    logits = logits_cpu.to(device) if logits_cpu is not None else None
    bias = bias_cpu.to(device) if bias_cpu is not None else None
    shared_w13 = shared_w13_cpu.to(device) if shared_w13_cpu is not None else None
    shared_w2 = shared_w2_cpu.to(device) if shared_w2_cpu is not None else None

    return x, w13, w2, topk_ids, topk_weights, logits, bias, shared_w13, shared_w2, ref


def check(torch, output, reference):
    if output.shape != reference.shape:
        raise RuntimeError(f'unexpected result shape {tuple(output.shape)}, expected {tuple(reference.shape)}')
    output = output.cpu().float()
    error = (output - reference).abs()
    matched = torch.isclose(output, reference, atol=0.05, rtol=0.05).float().mean().item()
    finite = bool(torch.isfinite(output).all())
    return {
        'correct': finite and matched >= 0.995,
        'finite': finite,
        'matched_ratio': matched,
        'max_abs_error': error.max().item() if finite else None,
    }


def benchmark(torch, call, warmup, repeat):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    events, wall = [], []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        before = time.perf_counter_ns()
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter_ns() - before) / 1000)
        elapsed = start.elapsed_time(end) * 1000
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f'invalid NPU Event duration: {elapsed}')
        events.append(elapsed)
    return {
        'event_median_us': statistics.median(events),
        'wall_median_us': statistics.median(wall),
        'event_samples_us': events,
        'wall_samples_us': wall,
    }


def profile_once(profiler, torch, call, directory):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
    ):
        call()
        torch.npu.synchronize()
    return [str(path) for path in sorted(directory.rglob('kernel_details.csv'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-tokens', type=int, default=1, help='1 for single-token decode, >1 for batch/prefill')
    parser.add_argument('--num-experts', type=int, default=256, help='DeepSeek-V3 default is 256')
    parser.add_argument('--topk', type=int, default=8, help='DeepSeek-V3 default is 8')
    parser.add_argument('--hidden-size', type=int, default=7168, help='DeepSeek-V3 default is 7168')
    parser.add_argument('--intermediate-size', type=int, default=2048, help='DeepSeek-V3 default is 2048')
    parser.add_argument('--n-group', type=int, default=8)
    parser.add_argument('--k-group', type=int, default=4)
    parser.add_argument('--routed-scaling-factor', type=float, default=2.5)
    parser.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--combine-op', choices=('unpermute', 'finalize'), default='unpermute')
    parser.add_argument('--include-gating', action='store_true', help='Include npu_moe_gating_top_k in timed region')
    parser.add_argument('--include-shared', action='store_true', help='Include 1 DeepSeek shared expert in timed region')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--profile-dir', type=Path, help='new directory for separate one-call device profiles')
    parser.add_argument('--output', type=Path, default=Path('deepseek_moe_cann.json'))
    args = parser.parse_args()

    if args.num_tokens < 1 or args.num_experts < args.topk or args.repeat < 1 or args.warmup < 0:
        parser.error('require num_tokens >= 1, num_experts >= topk, repeat >= 1 and warmup >= 0')
    if args.include_gating and args.num_experts % args.n_group != 0:
        parser.error('require num_experts divisible by n_group when --include-gating is set')
    if args.profile_dir and args.profile_dir.exists():
        parser.error('profile-dir already exists; choose a fresh path')

    import torch
    import torch_npu

    device = torch.device('npu:0')
    torch.npu.set_device(device)
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    report = {
        'device': torch.npu.get_device_name(device),
        'torch': str(torch.__version__),
        'torch_npu': getattr(torch_npu, '__version__', None),
        'cann_path': os.environ.get('ASCEND_TOOLKIT_HOME') or os.environ.get('ASCEND_HOME_PATH'),
        'shape': {
            'num_tokens': args.num_tokens,
            'num_experts': args.num_experts,
            'topk': args.topk,
            'hidden_size': args.hidden_size,
            'intermediate_size': args.intermediate_size,
            'dtype': args.dtype,
            'combine_op': args.combine_op,
            'include_gating': args.include_gating,
            'include_shared': args.include_shared,
        },
        'seed': args.seed,
        'warmup': args.warmup,
        'repeat': args.repeat,
        'results': {},
    }
    print(f"device={report['device']} shape={report['shape']}", flush=True)
    print('[init] allocating weights on NPU and computing CPU FP32 reference...', flush=True)

    x, w13, w2, topk_ids, topk_weights, logits, bias, shared_w13, shared_w2, reference = build_inputs(
        torch,
        device,
        args.num_tokens,
        args.num_experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        dtype,
        args.seed,
        args.include_gating,
        args.include_shared,
        args.n_group,
        args.k_group,
        args.routed_scaling_factor,
    )

    active_num = args.num_tokens * args.topk
    row_idx_type = 1 if args.combine_op == 'unpermute' else 0

    def cann_call():
        if args.include_gating:
            weights, ids, _ = torch_npu.npu_moe_gating_top_k(
                x=logits,
                k=args.topk,
                bias=bias,
                k_group=args.k_group,
                group_count=args.n_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=args.routed_scaling_factor,
            )
            ids = ids.to(torch.int32)
            weights = weights.to(dtype)
        else:
            ids = topk_ids
            weights = topk_weights

        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            x,
            expert_idx=ids,
            scale=None,
            active_num=active_num,
            expert_capacity=-1,
            expert_num=args.num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, args.num_experts],
            quant_mode=-1,
            row_idx_type=row_idx_type,
        )

        gate_up = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[w13],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        act = torch_npu.npu_swiglu(gate_up, dim=-1)

        down = torch_npu.npu_grouped_matmul(
            x=[act],
            weight=[w2],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        if args.combine_op == 'unpermute':
            out = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=down,
                sorted_indices=expanded_row_idx,
                probs=weights,
            )
        else:
            out = torch_npu.npu_moe_finalize_routing(
                expanded_permuted_rows=down,
                skip1=None,
                skip2=None,
                bias=None,
                scales=weights,
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=ids,
                drop_pad_mode=0,
            )

        if args.include_shared:
            s_gate_up = torch.matmul(x, shared_w13)
            s_act = torch_npu.npu_swiglu(s_gate_up, dim=-1)
            out = out + torch.matmul(s_act, shared_w2)

        return out

    with torch.inference_mode():
        print('[cann] first call and CPU FP32 check', flush=True)
        result = {}
        try:
            result.update(check(torch, cann_call(), reference))
            if result['correct']:
                result.update(benchmark(torch, cann_call, args.warmup, args.repeat))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')

        if args.profile_dir and result.get('correct') and 'error' not in result:
            try:
                result['profile_csv'] = profile_once(torch_npu.profiler, torch, cann_call, args.profile_dir / 'cann')
            except (AttributeError, RuntimeError) as exc:
                result['profile_error'] = str(exc)

        report['results']['cann'] = result
        print('cann', {k: v for k, v in result.items() if not k.endswith('_samples_us')}, flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    print(f'Saved {args.output}', flush=True)
    return int(not result.get('correct') or 'error' in result)


if __name__ == '__main__':
    raise SystemExit(main())
