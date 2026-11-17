// Online causal softmax stage for a tiled FlashAttention-style forward pass.
// Independent implementation of the recurrence described by FlashAttention;
// CUDA reference: Dao-AILab/flash-attention flash_fwd_kernel.h, commit 9f9742c.
// The two matrix multiplies are supplied by the caller, not fused into this kernel.
#include <torch/extension.h>
#include <cstdint>
#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

constexpr int TILE = 128;
// Each row has its own 64-byte scalar D-cache line. Distinct AIV cores must
// never SetValue different elements in the same line.
constexpr uint32_t STATE_STRIDE = 16;

template <AscendC::HardEvent E>
__aicore__ inline void sync_stage(AscendC::TPipe& pipe)
{
    int32_t event = static_cast<int32_t>(pipe.FetchEventID(E));
    AscendC::SetFlag<E>(event);
    AscendC::WaitFlag<E>(event);
}

__global__ __vector__ void online_stage_kernel(GM_ADDR scores, GM_ADDR old_max,
    GM_ADDR old_sum, GM_ADDR probs, GM_ADDR alpha, GM_ADDR new_max, GM_ADDR new_sum,
    uint32_t rows, uint32_t rows_per_head, uint32_t key_start, float scale)
{
    AscendC::GlobalTensor<half> s, p;
    AscendC::GlobalTensor<float> om, os, a, nm, ns;
    s.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scores));
    p.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(probs));
    om.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(old_max));
    os.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(old_sum));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
    nm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(new_max));
    ns.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(new_sum));
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> values_buf, weights_buf, exp_buf;
    pipe.InitBuffer(values_buf, TILE * sizeof(float));
    pipe.InitBuffer(weights_buf, TILE * sizeof(half));
    pipe.InitBuffer(exp_buf, 64 * sizeof(float));
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_buf, old_m_buf, old_l_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> alpha_buf, max_buf, sum_buf;
    pipe.InitBuffer(input_buf, TILE * sizeof(half));
    pipe.InitBuffer(old_m_buf, 8 * sizeof(float));
    pipe.InitBuffer(old_l_buf, 8 * sizeof(float));
    pipe.InitBuffer(alpha_buf, STATE_STRIDE * sizeof(float));
    pipe.InitBuffer(max_buf, STATE_STRIDE * sizeof(float));
    pipe.InitBuffer(sum_buf, STATE_STRIDE * sizeof(float));
    auto input = input_buf.Get<half>();
    auto old_m = old_m_buf.Get<float>();
    auto old_l = old_l_buf.Get<float>();
    auto alpha_out = alpha_buf.Get<float>();
    auto max_out = max_buf.Get<float>();
    auto sum_out = sum_buf.Get<float>();
    auto values = values_buf.Get<float>();
    auto weights = weights_buf.Get<half>();
    auto scratch = exp_buf.Get<float>();
    for (uint32_t r = AscendC::GetBlockIdx(); r < rows; r += AscendC::GetBlockNum()) {
        const uint32_t query_pos = key_start + r % rows_per_head;
        // All GM reads use DMA: allocator reuse cannot leave stale scalar D-cache reads.
        sync_stage<AscendC::HardEvent::S_MTE2>(pipe);
        AscendC::DataCopy(input, s[static_cast<uint64_t>(r) * TILE], TILE);
        AscendC::DataCopy(old_m, om[(r / 8) * 8], 8);
        AscendC::DataCopy(old_l, os[(r / 8) * 8], 8);
        sync_stage<AscendC::HardEvent::MTE2_S>(pipe);
        float previous_max = old_m.GetValue(r % 8);
        float m = previous_max;
        for (uint32_t j = 0; j < TILE; ++j) {
            const float x = (key_start + j <= query_pos)
                ? static_cast<float>(input.GetValue(j)) * scale
                : -3.402823466e+38F;
            values.SetValue(j, x);
            if (x > m) m = x;
        }
        for (uint32_t j = 0; j < TILE; ++j) values.SetValue(j, values.GetValue(j) - m);
        for (uint32_t j = 0; j < 64; ++j) scratch.SetValue(j, 0.0f);
        scratch.SetValue(0, previous_max - m);
        sync_stage<AscendC::HardEvent::S_V>(pipe);
        AscendC::Exp(values, values, TILE);
        AscendC::Exp(scratch, scratch, 64);
        sync_stage<AscendC::HardEvent::V_S>(pipe);
        const float correction = scratch.GetValue(0);
        float l = old_l.GetValue(r % 8) * correction;
        for (uint32_t j = 0; j < TILE; ++j) {
            const float x = values.GetValue(j);
            l += x;
            weights.SetValue(j, static_cast<half>(x));
        }
        for (uint32_t j = 0; j < STATE_STRIDE; ++j) {
            alpha_out.SetValue(j, 0.0f);
            max_out.SetValue(j, 0.0f);
            sum_out.SetValue(j, 0.0f);
        }
        alpha_out.SetValue(0, correction);
        max_out.SetValue(0, m);
        sum_out.SetValue(0, l);
        sync_stage<AscendC::HardEvent::S_MTE3>(pipe);
        AscendC::DataCopy(p[static_cast<uint64_t>(r) * TILE], weights, TILE);
        const uint64_t state_offset = static_cast<uint64_t>(r) * STATE_STRIDE;
        AscendC::DataCopy(a[state_offset], alpha_out, STATE_STRIDE);
        AscendC::DataCopy(nm[state_offset], max_out, STATE_STRIDE);
        AscendC::DataCopy(ns[state_offset], sum_out, STATE_STRIDE);
        // Complete outgoing transfers before the next row reuses UB buffers.
        sync_stage<AscendC::HardEvent::MTE3_S>(pipe);
    }
}

std::vector<at::Tensor> online_stage(const at::Tensor& scores, const at::Tensor& old_max,
    const at::Tensor& old_sum, int64_t rows_per_head, int64_t key_start, double scale)
{
    TORCH_CHECK(scores.device().type() == c10::DeviceType::PrivateUse1 &&
        scores.device() == old_max.device() && scores.device() == old_sum.device(), "NPU tensors required");
    TORCH_CHECK(scores.scalar_type() == at::kHalf && old_max.scalar_type() == at::kFloat &&
        old_sum.scalar_type() == at::kFloat, "expected fp16 scores and fp32 state");
    TORCH_CHECK(scores.is_contiguous() && old_max.is_contiguous() && old_sum.is_contiguous(), "contiguous required");
    TORCH_CHECK(scores.dim() == 2 && scores.size(1) == TILE && rows_per_head > 0 && key_start >= 0 &&
        scores.size(0) > 0 && scores.size(0) % 8 == 0 && scores.size(0) % rows_per_head == 0 && old_max.numel() == scores.size(0) && old_sum.numel() == scores.size(0), "invalid shapes");
    auto probs = at::empty_like(scores);
    // NPU allocations must be cache-line aligned so adjacent padded rows do
    // not share a scalar cache line across cores.
    auto alpha = at::empty({scores.size(0), STATE_STRIDE}, old_max.options());
    auto new_max = at::empty_like(alpha);
    auto new_sum = at::empty_like(alpha);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(alpha.data_ptr()) % 64 == 0 &&
        reinterpret_cast<uintptr_t>(new_max.data_ptr()) % 64 == 0 &&
        reinterpret_cast<uintptr_t>(new_sum.data_ptr()) % 64 == 0, "state buffers must be 64-byte aligned");
    // This direct launch bypasses TorchNPU's OpCommand task queue. Drain queued
    // host submissions first so QK/state producers reach the ACL stream before
    // our consumer kernel. stream(false) is safe only with queued launch integration.
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    online_stage_kernel<<<24, nullptr, stream>>>(
        (uint8_t*)scores.data_ptr(), (uint8_t*)old_max.data_ptr(), (uint8_t*)old_sum.data_ptr(),
        (uint8_t*)probs.data_ptr(), (uint8_t*)alpha.data_ptr(), (uint8_t*)new_max.data_ptr(),
        (uint8_t*)new_sum.data_ptr(), scores.size(0), rows_per_head, key_start, scale);
    // Compact padded device state on the same stream for the next operation.
    return {probs, alpha.select(1, 0).contiguous(), new_max.select(1, 0).contiguous(),
            new_sum.select(1, 0).contiguous()};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("online_stage", &online_stage); }
