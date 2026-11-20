// Own mixed AIC/AIV schedule. Two Mmad products per KV tile, no vector GEMM.
#pragma once
#include "geometry.hpp"
#ifdef CUBE_FA_CPU_SIM
#include "sim/cube_tiles_mock.hpp"
#else
#include "cube_tiles.hpp"
#endif

namespace CubeFA {
template<uint32_t Rows, uint32_t Cols>
__aicore__ inline void MaskFutureScores(AscendC::LocalTensor<float> scores, uint32_t first_visible)
{
    constexpr uint32_t lanes = 64;  // One 256-byte FP32 vector repeat.
    constexpr uint8_t row_stride = Cols * sizeof(float) / 32;  // In datablocks.
    constexpr uint64_t all_lanes = ~uint64_t{0};
    constexpr float blocked_score = -3.402823466e+38F;

    if constexpr (Cols == 2 * lanes) {
        if (first_visible <= lanes) {
            const uint32_t full = lanes + 1 - first_visible;
            const uint8_t repeats = static_cast<uint8_t>(full < Rows ? full : Rows);
            // In these rows, all of the second 64-column segment is masked.
            uint64_t mask[2] = {all_lanes, 0};
            AscendC::Duplicate(scores[lanes], blocked_score, mask, repeats, 1, row_stride);
        }
    }

    for (uint32_t row = 0; row < Rows && first_visible + row < Cols; ++row) {
        const uint32_t visible = first_visible + row;
        const uint32_t lane = visible % lanes;
        if (lane == 0) continue;  // The fully masked segment was filled above.
        // Keep the segment base aligned and preserve its visible prefix.
        uint64_t mask[2] = {all_lanes << lane, 0};
        const uint32_t segment_base = visible - lane;
        AscendC::Duplicate(scores[row * Cols + segment_base], blocked_score, mask, 1, 1, row_stride);
    }
}

template<int Core,uint32_t BM,uint32_t BN>
__aicore__ inline void RunCore(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR output,
    GM_ADDR relay,uint32_t heads,uint32_t seq,uint32_t groups)
{
    using G=Geometry<BM,BN>;
    const uint32_t group=AscendC::GetBlockIdx()/AscendC::GetSubBlockNum();
    const uint32_t sub=AscendC::GetSubBlockIdx();
    const uint32_t tiles=seq/BM, tasks=heads*tiles;
    const uint64_t offset=uint64_t(group)*G::RelayBytes;
    AscendC::GlobalTensor<half> qgm,kgm,vgm,ogm,pgm;
    AscendC::GlobalTensor<float> sgm,dgm;
    qgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q),uint64_t(heads)*seq*D);
    kgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k),uint64_t(heads)*seq*D);
    vgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v),uint64_t(heads)*seq*D);
    ogm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output),uint64_t(heads)*seq*D);
    // The first score slot precedes P/PV; the remaining score slots follow them.
    sgm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(relay+offset),G::RelayBytes/sizeof(float));
    pgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(relay+offset+G::SBytes),BM*BN);
    dgm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(relay+offset+G::SBytes+G::PBytes),BM*D);
    constexpr uint32_t second_score=G::SecondScoreOffset/sizeof(float);
    AscendC::TPipe pipe;
    if constexpr(Core==AscendC::AIC) {
        constexpr uint32_t MaxK=BN>D?BN:D, MaxN=MaxK;
        constexpr uint32_t L0CSlotElements=BM*MaxN;
        static_assert(2*L0CSlotElements*sizeof(float)<=128*1024);
        AscendC::TBuf<AscendC::TPosition::A1> qB,pB,bB;
        AscendC::TBuf<AscendC::TPosition::A2> a0B;
        AscendC::TBuf<AscendC::TPosition::B2> b0B;
        AscendC::TBuf<AscendC::TPosition::CO1> c0B;
        pipe.InitBuffer(qB,BM*D*sizeof(half));
        pipe.InitBuffer(pB,BM*BN*sizeof(half));
        pipe.InitBuffer(bB,BN*D*sizeof(half));
        pipe.InitBuffer(a0B,BM*MaxK*sizeof(half));
        pipe.InitBuffer(b0B,BN*D*sizeof(half));
        pipe.InitBuffer(c0B,2*L0CSlotElements*sizeof(float));
        auto ql=qB.template Get<half>(); auto pl=pB.template Get<half>(); auto bl=bB.template Get<half>();
        auto a0=a0B.template Get<half>(); auto b0=b0B.template Get<half>(); auto c0=c0B.template Get<float>();
        // Reserve FIX_M and M_FIX event IDs 0 and 1 for these two L0C slots.
        // Each FIX return releases only the slot its copy read.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        int32_t accum_slot=0;
        for(uint32_t task=group;task<tasks;task+=groups) {
            const uint32_t head=task/tiles, qs=(task%tiles)*BM;
            const uint64_t base=uint64_t(head)*seq*D;
            constexpr int32_t step=BN;
            const int32_t first_key=(qs/BN)*BN;
            uint32_t slot=0;
            for (int32_t qk_key=first_key, pv_key=first_key+int32_t(QkLookaheadTiles)*step;
                 pv_key>=0; qk_key-=step, pv_key-=step, slot=slot+1==ScoreSlots ? 0 : slot+1) {
                if (qk_key>=0) {
                    const uint32_t score_offset=slot ? second_score+(slot-1)*BM*BN : 0;
                    Gemm<true>(pipe,qgm[base+uint64_t(qs)*D],kgm[base+uint64_t(qk_key)*D],
                        sgm[score_offset],ql,bl,a0,b0,c0[accum_slot*L0CSlotElements],
                        BM,BN,D,Catlass::layout::RowMajor(BM,D),Catlass::layout::ColumnMajor(D,BN,D),
                        qk_key==first_key,accum_slot);
                    accum_slot^=1;
                    if (slot==0) AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(QK_READY);
                    else if (slot==1) AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(QK_NEXT_READY);
                    else AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(QK_THIRD_READY);
                }
                // Prime up to three QKs, then consume PV two tiles behind QK.
                if (pv_key>first_key) continue;
                AscendC::CrossCoreWaitFlag(P_READY);
                Gemm<true>(pipe,pgm,vgm[base+uint64_t(pv_key)*D],dgm,pl,bl,a0,b0,
                    c0[accum_slot*L0CSlotElements],BM,D,BN,Catlass::layout::RowMajor(BM,BN),
                    Catlass::layout::RowMajor(BN,D),true,accum_slot);
                accum_slot^=1;
                AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(PV_READY);
                // Drain PV before reusing the consumed tile's score slot and the shared relays.
                AscendC::CrossCoreWaitFlag(TILE_DONE);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    } else {
        constexpr uint32_t R=G::Rows;
        AscendC::TBuf<AscendC::TPosition::VECCALC> sB,pB,dB,oB,ohB,mB,lB,alphaB,maxAB,maxBB,sumAB,sumBB;
        pipe.InitBuffer(sB,R*BN*4); pipe.InitBuffer(pB,R*BN*2);
        pipe.InitBuffer(dB,R*D*4); pipe.InitBuffer(oB,R*D*4); pipe.InitBuffer(ohB,R*D*2);
        pipe.InitBuffer(mB,R*4); pipe.InitBuffer(lB,R*4); pipe.InitBuffer(alphaB,R*4);
        pipe.InitBuffer(maxAB,R*4); pipe.InitBuffer(maxBB,R*4);
        pipe.InitBuffer(sumAB,R*4); pipe.InitBuffer(sumBB,R*4);
        auto s=sB.template Get<float>(); auto p=pB.template Get<half>();
        auto delta=dB.template Get<float>(); auto out=oB.template Get<float>(); auto oh=ohB.template Get<half>();
        auto m=mB.template Get<float>(); auto l=lB.template Get<float>(); auto alpha=alphaB.template Get<float>();
        auto maxA=maxAB.template Get<float>(); auto maxB=maxBB.template Get<float>();
        auto sumA=sumAB.template Get<float>(); auto sumB=sumBB.template Get<float>();
        for(uint32_t task=group;task<tasks;task+=groups) {
            const uint32_t head=task/tiles, qs=(task%tiles)*BM;
            const uint64_t base=uint64_t(head)*seq*D;
            AscendC::Duplicate(out,0.0f,R*D);
            AscendC::Duplicate(l,0.0f,R);
            AscendC::Duplicate(m,-3.402823466e+38F,R);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t slot=0;
            for(int32_t ks=(qs/BN)*BN;ks>=0;ks-=int32_t(BN),slot=slot+1==ScoreSlots ? 0 : slot+1) {
                if (slot==0) AscendC::CrossCoreWaitFlag(QK_READY);
                else if (slot==1) AscendC::CrossCoreWaitFlag(QK_NEXT_READY);
                else AscendC::CrossCoreWaitFlag(QK_THIRD_READY);
                const uint32_t score_offset=slot ? second_score+(slot-1)*BM*BN : 0;
                Sync<AscendC::HardEvent::V_MTE2>(pipe);
                AscendC::DataCopy(s,sgm[score_offset+sub*R*BN],R*BN);
                Sync<AscendC::HardEvent::MTE2_V>(pipe);
                AscendC::Muls(s,s,0.08838834764831845f,R*BN);
                AscendC::PipeBarrier<PIPE_V>();
                if(uint32_t(ks)==(qs/BN)*BN) {
                    MaskFutureScores<R,BN>(s, qs+sub*R-uint32_t(ks)+1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::WholeReduceMax(maxA,s,64,R,1,1,BN/8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                if constexpr(BN==128) {
                    AscendC::WholeReduceMax(maxB,s[64],64,R,1,1,BN/8,AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Max(maxA,maxA,maxB,R);
                }
                Sync<AscendC::HardEvent::V_S>(pipe);
                for(uint32_t r=0;r<R;++r) {
                    const float old=m.GetValue(r), block=maxA.GetValue(r);
                    const float next=old>block?old:block;
                    m.SetValue(r,next); alpha.SetValue(r,old-next);
                    AscendC::Adds(s[r*BN],s[r*BN],-next,BN);
                }
                Sync<AscendC::HardEvent::S_V>(pipe);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(s,s,R*BN); AscendC::Exp(alpha,alpha,R);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(sumA,s,64,R,1,1,BN/8);
                if constexpr(BN==128) {
                    AscendC::WholeReduceSum(sumB,s[64],64,R,1,1,BN/8);
                    AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(sumA,sumA,sumB,R);
                }
                AscendC::Cast(p,s,AscendC::RoundMode::CAST_RINT,R*BN);
                AscendC::Mul(l,l,alpha,R);
                AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(l,l,sumA,R);
                Sync<AscendC::HardEvent::V_MTE3>(pipe);
                AscendC::DataCopy(pgm[sub*R*BN],p,R*BN);
                AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(P_READY);
                AscendC::CrossCoreWaitFlag(PV_READY);
                Sync<AscendC::HardEvent::V_MTE2>(pipe);
                AscendC::DataCopy(delta,dgm[sub*R*D],R*D);
                Sync<AscendC::HardEvent::MTE2_V>(pipe);
                Sync<AscendC::HardEvent::V_S>(pipe);
                for(uint32_t r=0;r<R;++r) AscendC::Muls(out[r*D],out[r*D],alpha.GetValue(r),D);
                AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(out,out,delta,R*D);
                AscendC::CrossCoreSetFlag<0x2,PIPE_V>(TILE_DONE);
                AscendC::PipeBarrier<PIPE_V>();
            }
            Sync<AscendC::HardEvent::V_S>(pipe);
            for(uint32_t r=0;r<R;++r) AscendC::Muls(out[r*D],out[r*D],1.0f/l.GetValue(r),D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(oh,out,AscendC::RoundMode::CAST_RINT,R*D);
            Sync<AscendC::HardEvent::V_MTE3>(pipe);
            AscendC::DataCopy(ogm[base+uint64_t(qs+sub*R)*D],oh,R*D);
            Sync<AscendC::HardEvent::MTE3_V>(pipe);
        }
    }
}

#ifndef CUBE_FA_CPU_SIM
template<uint32_t BM,uint32_t BN>
__global__ __mix__(1,2) void fused_cube_attention(uint64_t sync_base,
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR out,GM_ADDR relay,uint32_t heads,uint32_t seq,uint32_t groups)
{
    AscendC::SetSyncBaseAddr(sync_base);
    RunCore<g_coreType,BM,BN>(q,k,v,out,relay,heads,seq,groups);
}
#endif
} // namespace CubeFA
