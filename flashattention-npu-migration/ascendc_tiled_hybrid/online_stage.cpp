// Online causal softmax stage for a tiled FlashAttention-style forward pass.
// Independent implementation of the recurrence described by FlashAttention;
// CUDA reference: Dao-AILab/flash-attention flash_fwd_kernel.h, commit 9f9742c.
// The two matrix multiplies are supplied by the caller, not fused into this kernel.
#include <torch/extension.h>
#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

constexpr int TILE = 128;

__global__ __aicore__ void online_stage_kernel(GM_ADDR scores, GM_ADDR old_max,
    GM_ADDR old_sum, GM_ADDR probs, GM_ADDR alpha, GM_ADDR new_max, GM_ADDR new_sum,
    uint32_t rows, uint32_t rows_per_head, uint32_t key_start, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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
    auto values = values_buf.Get<float>();
    auto weights = weights_buf.Get<half>();
    auto scratch = exp_buf.Get<float>();
    for (uint32_t r = AscendC::GetBlockIdx(); r < rows; r += AscendC::GetBlockNum()) {
        const uint32_t query_pos = key_start + r % rows_per_head;
        float previous_max = om.GetValue(r);
        float m = previous_max;
        for (uint32_t j = 0; j < TILE; ++j) {
            const float x = (key_start + j <= query_pos)
                ? static_cast<float>(s.GetValue(static_cast<uint64_t>(r) * TILE + j)) * scale
                : -3.402823466e+38F;
            values.SetValue(j, x);
            if (x > m) m = x;
        }
        for (uint32_t j = 0; j < TILE; ++j) values.SetValue(j, values.GetValue(j) - m);
        for (uint32_t j = 0; j < 64; ++j) scratch.SetValue(j, 0.0f);
        scratch.SetValue(0, previous_max - m);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp(values, values, TILE);
        AscendC::Exp(scratch, scratch, 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float correction = scratch.GetValue(0);
        float l = os.GetValue(r) * correction;
        for (uint32_t j = 0; j < TILE; ++j) {
            const float x = values.GetValue(j);
            l += x;
            weights.SetValue(j, static_cast<half>(x));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(p[static_cast<uint64_t>(r) * TILE], weights, TILE);
        a.SetValue(r, correction);
        nm.SetValue(r, m);
        ns.SetValue(r, l);
        AscendC::PipeBarrier<PIPE_ALL>();
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
        scores.size(0) % rows_per_head == 0 && old_max.numel() == scores.size(0) && old_sum.numel() == scores.size(0), "invalid shapes");
    auto probs = at::empty_like(scores);
    auto alpha = at::empty_like(old_max);
    auto new_max = at::empty_like(old_max);
    auto new_sum = at::empty_like(old_sum);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    online_stage_kernel<<<24, nullptr, stream>>>(
        (uint8_t*)scores.data_ptr(), (uint8_t*)old_max.data_ptr(), (uint8_t*)old_sum.data_ptr(),
        (uint8_t*)probs.data_ptr(), (uint8_t*)alpha.data_ptr(), (uint8_t*)new_max.data_ptr(),
        (uint8_t*)new_sum.data_ptr(), scores.size(0), rows_per_head, key_start, scale);
    return {probs, alpha, new_max, new_sum};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("online_stage", &online_stage); }
