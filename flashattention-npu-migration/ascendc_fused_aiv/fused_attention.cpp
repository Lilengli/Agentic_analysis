#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "fused_kernel.hpp"

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                   int64_t block_m = 16, int64_t block_n = 32, int64_t blocks = 48)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1 && q.device() == k.device() &&
        q.device() == v.device(), "q/k/v must be on the same NPU");
    TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "matching BNSD shapes required");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
        "FP16 required");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "inference only; backward is not implemented");
    TORCH_CHECK((block_m == 8 || block_m == 16) && (block_n == 32 || block_n == 64), "BM=8/16 and BN=32/64 required");
    TORCH_CHECK(blocks >= 1 && blocks <= 48, "blocks must be 1..48");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(2) <= 16384 &&
        q.size(2) % block_n == 0 && q.size(3) == 128, "positive S<=16384 divisible by BN, D=128 required");
    const int64_t heads64 = q.size(0) * q.size(1);
    TORCH_CHECK(heads64 <= std::numeric_limits<uint32_t>::max() / (q.size(2) / block_m), "too many query tiles");
    c10::OptionalDeviceGuard guard(q.device());
    auto out = at::empty_like(q);
    // Direct launch: drain framework submissions, avoiding the ordering defect
    // of stream(false) with a launch outside TorchNPU's OpCommand task queue.
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    const uint32_t heads = heads64;
    const uint32_t seq = q.size(2);
    auto qp = static_cast<uint8_t*>(q.data_ptr());
    auto kp = static_cast<uint8_t*>(k.data_ptr());
    auto vp = static_cast<uint8_t*>(v.data_ptr());
    auto op = static_cast<uint8_t*>(out.data_ptr());
    if (block_m == 8 && block_n == 32) {
        FusedAiv::fused_causal_attention<8,32><<<blocks, nullptr, stream>>>(qp,kp,vp,op,heads,seq);
    } else if (block_m == 8 && block_n == 64) {
        FusedAiv::fused_causal_attention<8,64><<<blocks, nullptr, stream>>>(qp,kp,vp,op,heads,seq);
    } else if (block_m == 16 && block_n == 32) {
        FusedAiv::fused_causal_attention<16,32><<<blocks, nullptr, stream>>>(qp,kp,vp,op,heads,seq);
    } else {
        FusedAiv::fused_causal_attention<16,64><<<blocks, nullptr, stream>>>(qp,kp,vp,op,heads,seq);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("block_m")=16, pybind11::arg("block_n")=32, pybind11::arg("blocks")=48);
    m.def("metadata", []() {
        pybind11::dict d;
        d["engine"] = "AIV only; no Cube";
        d["kernel_launches_per_call"] = 1;
        d["gm_intermediate_workspace_bytes"] = 0;
        d["ub_bytes_bm8_bn32"] = FusedAiv::UbBytes<8,32>();
        d["ub_bytes_bm16_bn32"] = FusedAiv::UbBytes<16,32>();
        d["ub_bytes_bm8_bn64"] = FusedAiv::UbBytes<8,64>();
        d["ub_bytes_bm16_bn64"] = FusedAiv::UbBytes<16,64>();
        return d;
    });
}
