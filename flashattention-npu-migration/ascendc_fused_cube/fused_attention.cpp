#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <pybind11/stl.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#ifdef CUBE_FA_SIM_ZERO_FFTS
#include <cstdlib>
#include <cstring>
#endif
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
    int64_t bm=64,int64_t bn=128,int64_t groups=24,std::optional<bool> prefetch_pv_v=std::nullopt,
    bool cross_query=false,bool trim_diagonal=false,bool vector_normalize=false,bool balance_queries=false)
{
    TORCH_CHECK(q.device().type()==c10::DeviceType::PrivateUse1 && q.device()==k.device() && q.device()==v.device(),"same NPU required");
    TORCH_CHECK(q.dim()==4 && q.sizes()==k.sizes() && q.sizes()==v.sizes(),"matching BNSD tensors required");
    TORCH_CHECK(q.scalar_type()==at::kHalf && k.scalar_type()==at::kHalf && v.scalar_type()==at::kHalf,"FP16 required");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),"contiguous tensors required");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),"forward inference only");
    const bool wide_stack=bm==128 && bn==1024;
    const bool stacked=(bm==128 && bn==512) || wide_stack;
    TORCH_CHECK(((bm==32 || bm==64 || bm==128) && (bn==64 || bn==128) && bm<=bn) ||
        stacked,"BM32/64/128 BN64/128 with BM<=BN, or BM128 with BN512/1024 required");
    // With diagonal trimming, every KV slice ends at or before the query block.
    const int64_t seq_alignment=stacked && trim_diagonal ? bm : bn;
    const int64_t max_seq=stacked && trim_diagonal ? (1<<20) : (1<<14);
    TORCH_CHECK(q.size(0)>0 && q.size(1)>0 && q.size(2)>0 && q.size(2)<=max_seq &&
        q.size(2)%seq_alignment==0 && q.size(3)==128,
        "S<=",max_seq," divisible by ",seq_alignment," and D128 required");
    const bool prefetch_v=prefetch_pv_v.value_or(stacked);
    TORCH_CHECK(!prefetch_v || stacked,"prefetch_pv_v requires BM128 with BN512/1024");
    TORCH_CHECK(!cross_query || prefetch_v,"cross_query requires BM128 with BN512/1024 and V prefetch");
    TORCH_CHECK(!trim_diagonal || cross_query,"trim_diagonal requires cross_query");
    TORCH_CHECK(!vector_normalize || trim_diagonal,"vector_normalize requires trim_diagonal");
    TORCH_CHECK(!balance_queries || trim_diagonal,"balance_queries requires trim_diagonal");
    TORCH_CHECK(!wide_stack || (balance_queries && !vector_normalize),
        "BN1024 requires balanced queries and per-row output normalization");
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
    bool allow_zero_sync=false;
#ifdef CUBE_FA_SIM_ZERO_FFTS
    const char* simulator=std::getenv("IS_SIMULATOR_ENV");
    const char* allow_zero=std::getenv("CUBE_FA_SIM_ALLOW_ZERO_FFTS");
    allow_zero_sync=simulator && std::strcmp(simulator,"true")==0 &&
        allow_zero && std::strcmp(allow_zero,"1")==0;
#endif
    TORCH_CHECK(error==ACL_SUCCESS && (sync_ptr!=nullptr || allow_zero_sync),
        "aclrtGetHardwareSyncAddr failed: ",error);
    const uint64_t sync_base=reinterpret_cast<uint64_t>(sync_ptr);
    auto qp=static_cast<uint8_t*>(q.data_ptr()); auto kp=static_cast<uint8_t*>(k.data_ptr());
    auto vp=static_cast<uint8_t*>(v.data_ptr()); auto op=static_cast<uint8_t*>(out.data_ptr());
    auto wp=static_cast<uint8_t*>(relay.data_ptr()); const uint32_t seq=q.size(2);
    auto mp=static_cast<uint8_t*>(mask.data_ptr());
    if(bm==32 && bn==64) CubeFA::fused_cube_attention<32,64><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==32 && bn==128) CubeFA::fused_cube_attention<32,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==64 && bn==64) CubeFA::fused_cube_attention<64,64><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==64 && bn==128) CubeFA::fused_cube_attention<64,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(bm==128 && bn==128) CubeFA::fused_cube_attention<128,128><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(wide_stack)
        CubeFA::fused_cube_attention<128,1024,true,true,true,false,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(balance_queries && vector_normalize)
        CubeFA::fused_cube_attention<128,512,true,true,true,true,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(balance_queries)
        CubeFA::fused_cube_attention<128,512,true,true,true,false,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(vector_normalize) CubeFA::fused_cube_attention<128,512,true,true,true,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(trim_diagonal) CubeFA::fused_cube_attention<128,512,true,true,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(cross_query) CubeFA::fused_cube_attention<128,512,true,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else if(prefetch_v) CubeFA::fused_cube_attention<128,512,true><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    else CubeFA::fused_cube_attention<128,512><<<groups,nullptr,stream>>>(sync_base,qp,kp,vp,op,wp,mp,heads,seq,groups);
    return out;
}

at::Tensor product(const at::Tensor& a,const at::Tensor& b,bool transpose_b) {
    TORCH_CHECK(a.device().type()==c10::DeviceType::PrivateUse1 && a.device()==b.device() &&
        a.scalar_type()==at::kHalf && b.scalar_type()==at::kHalf && a.is_contiguous() && b.is_contiguous(),
        "same-device contiguous NPU FP16 matrices required");
    TORCH_CHECK(a.dim()==2 && b.dim()==2,"2D matrices required");
    const int64_t m=a.size(0),k=a.size(1),n=transpose_b?b.size(0):b.size(1);
    TORCH_CHECK((m==32 || m==64 || m==128) && (k==64 || k==128) && (n==64 || n==128) &&
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
        pybind11::arg("block_m")=64,pybind11::arg("block_n")=128,pybind11::arg("blocks")=24,
        pybind11::arg("prefetch_pv_v")=pybind11::none(),pybind11::arg("cross_query")=false,
        pybind11::arg("trim_diagonal")=false,pybind11::arg("vector_normalize")=false,
        pybind11::arg("balance_queries")=false);
    m.def("metadata",[](){
        pybind11::dict d;
        d["engine"]="AIC Cube + two AIV per group; QK and PV use Mmad";
        d["kernel_launches_per_call"]=1;
        d["gm_relay_bytes_per_group_bm32_bn64"]=CubeFA::Geometry<32,64>::RelayBytes;
        d["gm_relay_bytes_per_group_bm32_bn128"]=CubeFA::Geometry<32,128>::RelayBytes;
        d["gm_relay_bytes_per_group_bm64_bn64"]=CubeFA::Geometry<64,64>::RelayBytes;
        d["gm_relay_bytes_per_group_bm64_bn128"]=CubeFA::Geometry<64,128>::RelayBytes;
        d["gm_relay_bytes_per_group_bm128_bn128"]=CubeFA::Geometry<128,128>::RelayBytes;
        d["gm_relay_bytes_per_group_bm128_bn512"]=CubeFA::Geometry<128,512>::RelayBytes;
        d["gm_relay_bytes_per_group_bm128_bn1024"]=CubeFA::Geometry<128,1024>::RelayBytes;
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
        d["stack512_score_row_reductions"]="three-stage BlockReduceMax and BlockReduceSum";
        d["stack512_vector_batch_rows"]=CubeFA::Geometry<128,512>::BatchRows;
        d["stack512_default_prefetch_pv_v"]=true;
        d["stack512_cube_kv_per_mmad"]=CubeFA::Geometry<128,512>::CubeKv;
        d["stack1024_vector_batch_rows"]=CubeFA::Geometry<128,1024>::BatchRows;
        d["stack1024_cube_kv_per_mmad"]=CubeFA::Geometry<128,1024>::CubeKv;
        d["output_rescale"]="broadcast alpha and multiply before PV_READY";
        d["softmax_lookahead_tiles"]=CubeFA::PvRelaySlots-1;
        d["online_state"]="FP32 max, denominator and accumulated output in AIV UB";
        d["primitive_library"]="CATLASS tile copies and Mmad only; no existing attention operator";
        return d;
    });
}
