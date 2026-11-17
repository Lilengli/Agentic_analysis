// Independent algorithm-level implementation guided by Dao-AILab FlashAttention
// flash_fwd_kernel.h at 9f9742c80e69b3ca28a231c26ef931e98c2c3826.
// This vector-only variant retains intermediates in UB; it does not use Cube.
#pragma once
#include "kernel_operator.h"
#include <cstdint>

namespace FusedAiv {
constexpr uint32_t D = 128;

template <AscendC::HardEvent E>
__aicore__ inline void Sync(AscendC::TPipe& pipe) {
    auto id = static_cast<int32_t>(pipe.FetchEventID(E));
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

template <uint32_t BM, uint32_t BN>
constexpr uint32_t UbBytes() {
    return 2 * BM * D * sizeof(half) + 2 * BM * D * sizeof(float)
        + 2 * BN * D * sizeof(half) + 3 * BN * D * sizeof(float)
        + (2 * BN + BN + BN * 8 + 2 * BM) * sizeof(float);
}

template <uint32_t BM, uint32_t BN>
__global__ __vector__ void fused_causal_attention(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, uint32_t heads, uint32_t seq)
{
    static_assert((BM == 8 || BM == 16) && (BN == 32 || BN == 64));
    static_assert(BN % BM == 0);
    static_assert(UbBytes<BM, BN>() <= 180 * 1024, "UB budget exceeded");
    AscendC::GlobalTensor<half> qgm, kgm, vgm, ogm;
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), uint64_t(heads) * seq * D);
    kgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), uint64_t(heads) * seq * D);
    vgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), uint64_t(heads) * seq * D);
    ogm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), uint64_t(heads) * seq * D);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qhB, qfB, khB, vhB, kfB, vfB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prodB, probB, tmpB, broadB, outB, ohB, maxB, sumB;
    pipe.InitBuffer(qhB, BM * D * sizeof(half));
    pipe.InitBuffer(qfB, BM * D * sizeof(float));
    pipe.InitBuffer(khB, BN * D * sizeof(half));
    pipe.InitBuffer(vhB, BN * D * sizeof(half));
    pipe.InitBuffer(kfB, BN * D * sizeof(float));
    pipe.InitBuffer(vfB, BN * D * sizeof(float));
    pipe.InitBuffer(prodB, BN * D * sizeof(float));
    pipe.InitBuffer(probB, 2 * BN * sizeof(float));
    pipe.InitBuffer(tmpB, BN * sizeof(float));
    pipe.InitBuffer(broadB, BN * 8 * sizeof(float));
    pipe.InitBuffer(outB, BM * D * sizeof(float));
    pipe.InitBuffer(ohB, BM * D * sizeof(half));
    pipe.InitBuffer(maxB, BM * sizeof(float));
    pipe.InitBuffer(sumB, BM * sizeof(float));
    auto qh = qhB.Get<half>(); auto qf = qfB.Get<float>();
    auto kh = khB.Get<half>(); auto vh = vhB.Get<half>();
    auto kf = kfB.Get<float>(); auto vf = vfB.Get<float>();
    auto product = prodB.Get<float>(); auto prob = probB.Get<float>();
    auto tmp = tmpB.Get<float>(); auto broad = broadB.Get<float>();
    auto out = outB.Get<float>(); auto oh = ohB.Get<half>();
    auto maxima = maxB.Get<float>(); auto sums = sumB.Get<float>();
    const uint32_t tiles = seq / BM;
    const uint32_t tasks = heads * tiles;
    for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += AscendC::GetBlockNum()) {
        const uint32_t head = task / tiles;
        const uint32_t qs = (task % tiles) * BM;
        const uint64_t base = uint64_t(head) * seq * D;
        Sync<AscendC::HardEvent::V_MTE2>(pipe);
        AscendC::DataCopy(qh, qgm[base + uint64_t(qs) * D], BM * D);
        Sync<AscendC::HardEvent::MTE2_V>(pipe);
        AscendC::Cast(qf, qh, AscendC::RoundMode::CAST_NONE, BM * D);
        AscendC::Duplicate(out, 0.0f, BM * D);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < BM; ++r) {
            maxima.SetValue(r, -3.402823466e+38F);
            sums.SetValue(r, 0.0f);
        }
        // As in the CUDA row-block forward: visit the causal diagonal first,
        // then unmasked earlier key blocks, retaining Q/O/m/l throughout.
        for (int32_t ks = (qs / BN) * BN; ks >= 0; ks -= static_cast<int32_t>(BN)) {
            Sync<AscendC::HardEvent::V_MTE2>(pipe);
            AscendC::DataCopy(kh, kgm[base + uint64_t(ks) * D], BN * D);
            AscendC::DataCopy(vh, vgm[base + uint64_t(ks) * D], BN * D);
            Sync<AscendC::HardEvent::MTE2_V>(pipe);
            AscendC::Cast(kf, kh, AscendC::RoundMode::CAST_NONE, BN * D);
            AscendC::Cast(vf, vh, AscendC::RoundMode::CAST_NONE, BN * D);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < BM; ++r) {
                // Vector implementation of QK: two 64-float halves, BN rows
                // in parallel. src1RepStride=0 reuses this query's 128 floats.
                AscendC::Mul(product, kf, qf[r * D], 64, BN, {1,1,1,16,16,0});
                AscendC::Mul(product[64], kf[64], qf[r * D + 64], 64, BN, {1,1,1,16,16,0});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(prob, product, 64, BN, 1, 1, 16);
                AscendC::WholeReduceSum(tmp, product[64], 64, BN, 1, 1, 16);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(prob, prob, tmp, BN);
                Sync<AscendC::HardEvent::V_S>(pipe);
                const float old_m = maxima.GetValue(r);
                float new_m = old_m;
                for (uint32_t j = 0; j < BN; ++j) {
                    const float x = uint32_t(ks) + j <= qs + r
                        ? prob.GetValue(j) * 0.08838834764831845f : -3.402823466e+38F;
                    prob.SetValue(j, x);
                    if (x > new_m) new_m = x;
                }
                for (uint32_t j = 0; j < BN; ++j) prob.SetValue(j, prob.GetValue(j) - new_m);
                for (uint32_t j = BN; j < 2 * BN; ++j) prob.SetValue(j, 0.0f);
                prob.SetValue(BN, old_m - new_m);
                Sync<AscendC::HardEvent::S_V>(pipe);
                AscendC::Exp(prob, prob, 2 * BN);
                Sync<AscendC::HardEvent::V_S>(pipe);
                const float alpha = prob.GetValue(BN);
                float new_l = sums.GetValue(r) * alpha;
                for (uint32_t j = 0; j < BN; ++j) new_l += prob.GetValue(j);
                maxima.SetValue(r, new_m);
                sums.SetValue(r, new_l);
                Sync<AscendC::HardEvent::S_V>(pipe);
                // Broadcast each p[j] into one 32-byte block. A zero source
                // block stride broadcasts that block across a 64-float half.
                AscendC::Brcb(broad, prob, BN / 8, {1,8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, vf, broad, 64, BN, {1,1,0,16,16,1});
                AscendC::Mul(product[64], vf[64], broad, 64, BN, {1,1,0,16,16,1});
                AscendC::PipeBarrier<PIPE_V>();
                // Tree reduction along keys leaves the 128-dimensional PV
                // result in product[0:D], without an off-chip intermediate.
                for (uint32_t n = BN / 2; n > 0; n /= 2) {
                    AscendC::Add(product, product, product[n * D], n * D);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Muls(out[r * D], out[r * D], alpha, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(out[r * D], out[r * D], product, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        for (uint32_t r = 0; r < BM; ++r) {
            const float inv_l = 1.0f / sums.GetValue(r);
            AscendC::Muls(out[r * D], out[r * D], inv_l, D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(oh, out, AscendC::RoundMode::CAST_RINT, BM * D);
        Sync<AscendC::HardEvent::V_MTE3>(pipe);
        // This is the only GM write in the kernel: one final output per Q tile.
        AscendC::DataCopy(ogm[base + uint64_t(qs) * D], oh, BM * D);
        Sync<AscendC::HardEvent::MTE3_S>(pipe);
    }
}
} // namespace FusedAiv
