// Experimental algorithm-level port of the causal forward path in
// Dao-AILab/flash-attention csrc/flash_attn/src/flash_fwd_kernel.h, commit
// 9f9742c80e69b3ca28a231c26ef931e98c2c3826. See UPSTREAM_LICENSE.
// This correctness-first AIV/scalar prototype is NOT the CUDA tiled-MMA
// implementation and is not a performance replacement for FlashAttention.
#include <torch/extension.h>
#include "kernel_operator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

constexpr uint32_t D = 128;
constexpr uint32_t KV_TILE = 32;
constexpr uint32_t EXP_LANES = 64;

__global__ __aicore__ void flash_ascendc_probe(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                               GM_ADDR out, uint32_t heads, uint32_t seq)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<half> qgm, kgm, vgm, ogm;
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q));
    kgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k));
    vgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v));
    ogm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buffer, out_buffer;
    pipe.InitBuffer(exp_buffer, EXP_LANES * sizeof(float));
    pipe.InitBuffer(out_buffer, D * sizeof(half));
    auto exp_local = exp_buffer.Get<float>();
    auto out_local = out_buffer.Get<half>();

    const uint32_t total_rows = heads * seq;  // host enforces batch=1
    for (uint32_t task = AscendC::GetBlockIdx(); task < total_rows; task += AscendC::GetBlockNum()) {
        const uint32_t head = task / seq;
        const uint32_t row = task % seq;
        const uint64_t base = static_cast<uint64_t>(head) * seq * D;
        float query[D];
        float accum[D];
        for (uint32_t d = 0; d < D; ++d) {
            query[d] = static_cast<float>(qgm.GetValue(base + static_cast<uint64_t>(row) * D + d));
            accum[d] = 0.0f;
        }
        float running_max = -3.402823466e+38F;
        float running_sum = 0.0f;
        for (uint32_t start = 0; start <= row; start += KV_TILE) {
            uint32_t count = row + 1 - start;
            if (count > KV_TILE) count = KV_TILE;
            float scores[KV_TILE];
            float next_max = running_max;
            for (uint32_t j = 0; j < count; ++j) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; ++d) {
                    dot += query[d] * static_cast<float>(kgm.GetValue(base + static_cast<uint64_t>(start + j) * D + d));
                }
                scores[j] = dot * 0.08838834764831845f;  // 1/sqrt(128)
                if (scores[j] > next_max) next_max = scores[j];
            }
            // One vector Exp computes the weights and previous-state rescale.
            for (uint32_t j = 0; j < EXP_LANES; ++j) exp_local.SetValue(j, -80.0f);
            for (uint32_t j = 0; j < count; ++j) exp_local.SetValue(j, scores[j] - next_max);
            exp_local.SetValue(KV_TILE, running_max - next_max);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(exp_local, exp_local, EXP_LANES);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float correction = exp_local.GetValue(KV_TILE);
            float block_sum = 0.0f;
            for (uint32_t d = 0; d < D; ++d) accum[d] *= correction;
            for (uint32_t j = 0; j < count; ++j) {
                const float weight = exp_local.GetValue(j);
                block_sum += weight;
                for (uint32_t d = 0; d < D; ++d) {
                    accum[d] += weight * static_cast<float>(vgm.GetValue(base + static_cast<uint64_t>(start + j) * D + d));
                }
            }
            running_sum = running_sum * correction + block_sum;
            running_max = next_max;
        }
        for (uint32_t d = 0; d < D; ++d) out_local.SetValue(d, static_cast<half>(accum[d] / running_sum));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(ogm[base + static_cast<uint64_t>(row) * D], out_local, D);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1 && q.device() == k.device() && q.device() == v.device(),
                "q/k/v must be on the same NPU");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "q/k/v must be float16");
    TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "shapes must match and be BNSD");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(q.size(0) == 1 && q.size(1) >= 1 && q.size(1) <= 2 && q.size(2) >= 1 && q.size(2) <= 128 && q.size(3) == D,
                "prototype supports B=1, H=1..2, S=1..128, D=128");
    auto out = at::empty_like(q);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    flash_ascendc_probe<<<24, nullptr, stream>>>(
        static_cast<uint8_t*>(q.data_ptr()), static_cast<uint8_t*>(k.data_ptr()),
        static_cast<uint8_t*>(v.data_ptr()), static_cast<uint8_t*>(out.data_ptr()),
        static_cast<uint32_t>(q.size(1)), static_cast<uint32_t>(q.size(2)));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
