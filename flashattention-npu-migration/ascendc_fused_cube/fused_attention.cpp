#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "mixed_kernel.hpp"

namespace {
at::Tensor GetCausalMask(c10::Device device,int64_t bm,int64_t bn)
{
    using Key=std::tuple<c10::DeviceIndex,int64_t,int64_t>;
    static std::mutex mutex;
    static std::map<Key,at::Tensor> masks;
    const Key key{device.index(),bm,bn};
    const std::lock_guard<std::mutex> lock(mutex);
    const auto existing=masks.find(key);
    if (existing!=masks.end()) return existing->second;

    auto cpu=at::empty({2*CubeFA::MaskWordsPerAiv(bn)},at::TensorOptions().device(at::kCPU).dtype(at::kInt));
    CubeFA::PackCausalMasks(reinterpret_cast<uint32_t*>(cpu.data_ptr<int32_t>()),bm,bn);
    auto mask=cpu.to(device,at::kInt,/*non_blocking=*/false,/*copy=*/false);
    // Publish only after the initial upload, so other NPU streams can use it.
    c10_npu::getCurrentNPUStream().synchronize();
    return masks.emplace(key,mask).first->second;
}
} // namespace

at::Tensor forward(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    int64_t bm=64,int64_t bn=128,int64_t groups=24)
{
    TORCH_CHECK(q.device().type()==c10::DeviceType::PrivateUse1 && q.device()==k.device() && q.device()==v.device(),"same NPU required");
    TORCH_CHECK(q.dim()==4 && q.sizes()==k.sizes() && q.sizes()==v.sizes(),"matching BNSD tensors required");
    TORCH_CHECK(q.scalar_type()==at::kHalf && k.scalar_type()==at::kHalf && v.scalar_type()==at::kHalf,"FP16 required");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),"contiguous tensors required");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),"forward inference only");
    TORCH_CHECK(bm>0 && bm<=128 && bn>0 && bn<=256 && CubeFA::IsSupportedTile(bm,bn),
        "require BM32/64/128 with BN64/128 and BM<=BN, or BM64/BN256");
    TORCH_CHECK(q.size(0)>0 && q.size(1)>0 && q.size(2)>0 && q.size(2)<=16384 && q.size(2)%bn==0 && q.size(3)==128,"S<=16384 divisible by BN and D128 required");
    TORCH_CHECK(groups>0 && groups<=24,"groups must be 1..24");
    const int64_t heads=q.size(0)*q.size(1);
    TORCH_CHECK(heads<=std::numeric_limits<uint32_t>::max()/(q.size(2)/bm),"too many tasks");
    c10::OptionalDeviceGuard guard(q.device());
    const auto mask=GetCausalMask(q.device(),bm,bn);
    auto out=at::empty_like(q);
    const int64_t per_group=CubeFA::RelayBytesFor(bm,bn);
    // Fixed-size per-group relay; independent of S. Same-stream allocator
    // lifetime is safe because raw launch is submitted before tensor release.
    auto relay=at::empty({groups*per_group},q.options().dtype(at::kByte));
    auto stream=c10_npu::getCurrentNPUStream().stream(true);
    void* sync_ptr=nullptr;
    const auto error=aclrtGetHardwareSyncAddr(&sync_ptr);
    TORCH_CHECK(error==ACL_SUCCESS && sync_ptr!=nullptr,"aclrtGetHardwareSyncAddr failed: ",error);
    const uint64_t sync_base=reinterpret_cast<uint64_t>(sync_ptr);
    auto qp=static_cast<uint8_t*>(q.data_ptr()); auto kp=static_cast<uint8_t*>(k.data_ptr());
    auto vp=static_cast<uint8_t*>(v.data_ptr()); auto op=static_cast<uint8_t*>(out.data_ptr());
    auto wp=static_cast<uint8_t*>(relay.data_ptr()); const uint32_t seq=q.size(2);
    auto mp=static_cast<uint8_t*>(mask.data_ptr());
    if(bm==32 && bn==64) CubeFA::fused_cube_attention<32,64><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==32 && bn==128) CubeFA::fused_cube_attention<32,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==64 && bn==64) CubeFA::fused_cube_attention<64,64><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==64 && bn==128) CubeFA::fused_cube_attention<64,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==64 && bn==256) CubeFA::fused_cube_attention<64,256><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else CubeFA::fused_cube_attention<128,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    return out;
}

at::Tensor product(const at::Tensor& a,const at::Tensor& b,bool transpose_b) {
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && a.device()==b.device() &&
        a.scalar_type()==at::kHalf && b.scalar_type()==at::kHalf && a.is_contiguous() && b.is_contiguous(),
        "same-device contiguous NPU FP16 matrices required");
    TORCH_CHECK(a.dim()==2 && b.dim()==2,"2D matrices required");
    const int64_t m=a.size(0),k=a.size(1),n=transpose_b?b.size(0):b.size(1);
    const bool standard=(m==32 || m==64 || m==128) && (k==64 || k==128) && (n==64 || n==128);
    const bool wide=m==64 && (transpose_b ? k==128 && n==256 : k==256 && n==128);
    TORCH_CHECK((standard || wide) &&
        (transpose_b?b.size(1):b.size(0))==k,"unsupported test shape");
    c10::OptionalDeviceGuard guard(a.device());
    auto c=at::empty({m,n},a.options().dtype(at::kFloat));
    auto stream=c10_npu::getCurrentNPUStream().stream(true);
    auto ap=static_cast<uint8_t*>(a.data_ptr()); auto bp=static_cast<uint8_t*>(b.data_ptr());
    auto cp=static_cast<uint8_t*>(c.data_ptr());
    if(transpose_b) CubeFA::cube_product_probe<true><<<1,nullptr,stream>>>(ap,bp,cp,m,n,k);
    else CubeFA::cube_product_probe<false><<<1,nullptr,stream>>>(ap,bp,cp,m,n,k);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("product",&product,pybind11::arg("a"),pybind11::arg("b"),pybind11::arg("transpose_b")=false);
    m.def("forward",&forward,pybind11::arg("q"),pybind11::arg("k"),pybind11::arg("v"),
        pybind11::arg("block_m")=64,pybind11::arg("block_n")=128,pybind11::arg("blocks")=24);
    m.def("metadata",[](){
        pybind11::dict d;
        d["engine"]="AIC Cube + two AIV per group; QK and PV use Mmad";
        d["kernel_launches_per_call"]=1;
        d["gm_relay_bytes_per_group_bm32_bn64"]=CubeFA::Geometry<32,64>::RelayBytes;
        d["gm_relay_bytes_per_group_bm32_bn128"]=CubeFA::Geometry<32,128>::RelayBytes;
        d["gm_relay_bytes_per_group_bm64_bn64"]=CubeFA::Geometry<64,64>::RelayBytes;
        d["gm_relay_bytes_per_group_bm64_bn128"]=CubeFA::Geometry<64,128>::RelayBytes;
        d["gm_relay_bytes_per_group_bm64_bn256"]=CubeFA::Geometry<64,256>::RelayBytes;
        d["gm_relay_bytes_per_group_bm128_bn128"]=CubeFA::Geometry<128,128>::RelayBytes;
        d["gm_intermediates"]="FP32 score ring, three FP16 probability slots and three FP32 PV slots per group";
        d["gm_score_slots"]=CubeFA::ScoreSlots;
        d["gm_pv_slots"]=CubeFA::PvRelaySlots;
        d["qk_lookahead_tiles"]=CubeFA::QkLookaheadTiles;
        d["aiv_score_ub_slots"]=CubeFA::UbScoreSlots;
        d["causal_mask"]="cached packed GM bitmap prefetched to UB for vector Select";
        d["score_centering"]="Brcb and strided vector Sub";
        d["online_max_update"]="vector Max/Sub with alternating UB views";
        d["score_row_max"]="two-stage BlockReduceMax";
        d["score_row_sum"]="two-stage BlockReduceSum";
        d["output_rescale"]="broadcast alpha and multiply before PV_READY";
        d["softmax_lookahead_tiles"]=CubeFA::PvRelaySlots-1;
        d["online_state"]="FP32 max, denominator and accumulated output in AIV UB";
        d["primitive_library"]="CATLASS tile copies and Mmad only; no existing attention operator";
        return d;
    });
}
