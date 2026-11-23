// Own mixed AIC/AIV schedule. Two Mmad products per KV tile, no vector GEMM.
#pragma once
#include "geometry.hpp"
#ifdef CUBE_FA_CPU_SIM
#include "sim/cube_tiles_mock.hpp"
#else
#include "cube_tiles.hpp"
#endif

namespace CubeFA {
__aicore__ inline uint16_t ScoreReadyFlag(uint32_t slot)
{
    return slot==0 ? QK_READY : QK_EXTRA_READY_BASE+slot-1;
}

__aicore__ inline uint16_t RelayFlag(uint16_t flag,uint32_t slot)
{
    return flag+slot*RelayFlagStride;
}

template<uint32_t R,uint32_t BN>
__aicore__ inline void PublishP(AscendC::GlobalTensor<half> pgm,AscendC::LocalTensor<half> p,
    uint32_t slot,uint32_t sub,AscendC::TPipe& pipe)
{
    Sync<AscendC::HardEvent::V_MTE3>(pipe);
    AscendC::DataCopy(pgm[(slot*2+sub)*R*BN],p,R*BN);
    AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RelayFlag(P_READY,slot));
}

template<uint32_t Rows,uint32_t BN>
__aicore__ inline void ReduceRowMax64(AscendC::LocalTensor<float> row_max,
    AscendC::LocalTensor<float> scores, AscendC::LocalTensor<float> partial)
{
    constexpr int32_t BlockElements=32/sizeof(float), RepeatElements=256/sizeof(float);
    // Each row's first reduction produces eight maxima in one data block.
    AscendC::BlockReduceMax(partial,scores,Rows,RepeatElements,1,1,BN/BlockElements);
    AscendC::PipeBarrier<PIPE_V>();
    // One repeat now covers the partial maxima from eight consecutive rows.
    AscendC::BlockReduceMax(row_max,partial,Rows/BlockElements,RepeatElements,1,1,8);
}

template<uint32_t Rows,uint32_t BN>
__aicore__ inline void ReduceRowSum64(AscendC::LocalTensor<float> row_sum,
    AscendC::LocalTensor<float> scores, AscendC::LocalTensor<float> partial)
{
    constexpr int32_t BlockElements=32/sizeof(float), RepeatElements=256/sizeof(float);
    // Pack each row's eight partial sums into one data block.
    AscendC::BlockReduceSum(partial,scores,Rows,RepeatElements,1,1,BN/BlockElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum(row_sum,partial,Rows/BlockElements,RepeatElements,1,1,8);
}

template<uint32_t Rows,uint32_t BN,bool ApplyCausalMask>
__aicore__ inline void PrepareScoreTile(AscendC::LocalTensor<float> s,
    AscendC::GlobalTensor<float> scores, AscendC::LocalTensor<float> maxA,
    AscendC::LocalTensor<float> maxB, AscendC::LocalTensor<float> row_blocks,
    uint32_t gm_slot, AscendC::LocalTensor<uint32_t> causal_mask)
{
    AscendC::CrossCoreWaitFlag(ScoreReadyFlag(gm_slot));
    AscendC::DataCopy(s,scores,Rows*BN);
    // Use distinct MTE2_V events for score and PV copies, which can both be queued.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Muls(s,s,0.08838834764831845f,Rows*BN);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (ApplyCausalMask) {
        AscendC::Select(s,causal_mask,s,-3.402823466e+38F,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,Rows*BN);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // The preceding barrier has released the row-block scratch used by output rescale.
    ReduceRowMax64<Rows,BN>(maxA,s,row_blocks);
    if constexpr(BN==128) {
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRowMax64<Rows,BN>(maxB,s[64],row_blocks);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(maxA,maxA,maxB,Rows);
    }
}

template<uint32_t R,uint32_t BN>
__aicore__ inline void CompleteSoftmax(AscendC::LocalTensor<float> s, AscendC::LocalTensor<half> p,
    AscendC::LocalTensor<float>& m, AscendC::LocalTensor<float> l, AscendC::LocalTensor<float> alpha,
    AscendC::LocalTensor<float>& maxA, AscendC::LocalTensor<float> sumA,
    AscendC::LocalTensor<float> sumB, AscendC::LocalTensor<float> row_blocks)
{
    constexpr uint32_t Fp32Block=32/sizeof(float), Fp32Repeat=256/sizeof(float);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(maxA,maxA,m,R);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(alpha,m,maxA,R);
    // Keep the updated maxima by exchanging the two UB views.
    // The old running buffer becomes scratch for the next reduction.
    auto previous_m=m;
    m=maxA;
    maxA=previous_m;
    AscendC::Brcb(row_blocks,m,R/Fp32Block,AscendC::BrcbRepeatParams(1,8));
    AscendC::PipeBarrier<PIPE_V>();
    // Reuse one 32-byte max block across each 64-column repeat, then advance a row.
    for (uint32_t col=0;col<BN;col+=Fp32Repeat) {
        AscendC::Sub(s[col],s[col],row_blocks,uint64_t{Fp32Repeat},R,
            AscendC::BinaryRepeatParams(1,1,0,BN/Fp32Block,BN/Fp32Block,1));
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(s,s,R*BN); AscendC::Exp(alpha,alpha,R);
    AscendC::PipeBarrier<PIPE_V>();
    // The preceding barrier releases the scratch used to broadcast row maxima.
    ReduceRowSum64<R,BN>(sumA,s,row_blocks);
    if constexpr(BN==128) {
        AscendC::PipeBarrier<PIPE_V>();
        ReduceRowSum64<R,BN>(sumB,s[64],row_blocks);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sumA,sumA,sumB,R);
    }
    AscendC::Cast(p,s,AscendC::RoundMode::CAST_RINT,R*BN);
    AscendC::Mul(l,l,alpha,R);
    AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(l,l,sumA,R);
}

template<int Core,uint32_t BM,uint32_t BN>
__aicore__ inline void RunCore(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR output,
    GM_ADDR relay,GM_ADDR causal_masks,uint32_t heads,uint32_t seq,uint32_t groups)
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
    pgm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(relay+offset+G::SBytes),PvRelaySlots*BM*BN);
    dgm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(relay+offset+G::SBytes+PvRelaySlots*G::PBytes),
        PvRelaySlots*BM*D);
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
            uint32_t slot=0, issued_pv=0;
            for (int32_t qk_key=first_key, pv_key=first_key+int32_t(QkLookaheadTiles)*step;
                 pv_key>=0; qk_key-=step, pv_key-=step, slot=slot+1==ScoreSlots ? 0 : slot+1) {
                if (qk_key>=0) {
                    const uint32_t score_offset=slot ? second_score+(slot-1)*BM*BN : 0;
                    Gemm<true>(pipe,qgm[base+uint64_t(qs)*D],kgm[base+uint64_t(qk_key)*D],
                        sgm[score_offset],ql,bl,a0,b0,c0[accum_slot*L0CSlotElements],
                        BM,BN,D,Catlass::layout::RowMajor(BM,D),Catlass::layout::ColumnMajor(D,BN,D),
                        qk_key==first_key,accum_slot);
                    accum_slot^=1;
                    AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(ScoreReadyFlag(slot));
                }
                // Prime the score queue before the first PV.
                if (pv_key>first_key) continue;
                const uint32_t pv_slot=issued_pv%PvRelaySlots;
                if (issued_pv>=PvRelaySlots) AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,pv_slot));
                AscendC::CrossCoreWaitFlag(RelayFlag(P_READY,pv_slot));
                Gemm<true>(pipe,pgm[pv_slot*BM*BN],vgm[base+uint64_t(pv_key)*D],dgm[pv_slot*BM*D],
                    pl,bl,a0,b0,c0[accum_slot*L0CSlotElements],BM,D,BN,Catlass::layout::RowMajor(BM,BN),
                    Catlass::layout::RowMajor(BN,D),true,accum_slot);
                accum_slot^=1;
                AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(RelayFlag(PV_READY,pv_slot));
                ++issued_pv;
            }
            const uint32_t first_pending=issued_pv>PvRelaySlots ? issued_pv-PvRelaySlots : 0;
            for(uint32_t tile=first_pending;tile<issued_pv;++tile)
                AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,tile%PvRelaySlots));
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    } else {
        constexpr uint32_t R=G::Rows;
        constexpr uint32_t ScoreElements=R*BN;
        constexpr uint32_t Fp32Block=32/sizeof(float), Fp32Repeat=256/sizeof(float);
        static_assert(R%Fp32Block==0 && BN%Fp32Repeat==0 && D%Fp32Repeat==0);
        static_assert(BN<=D);  // The final-output staging buffer also holds a P tile.
        static_assert(G::UbBytes+SelectScratchBytes<=192*1024-256);
        AscendC::TBuf<AscendC::TPosition::VECCALC> sB,pB,dB,oB,ohB,mB,lB,alphaB,maxAB,maxBB,sumAB,sumBB,maskB;
        AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxBlockB;
        pipe.InitBuffer(sB,UbScoreSlots*ScoreElements*sizeof(float)); pipe.InitBuffer(pB,R*BN*2);
        pipe.InitBuffer(dB,R*D*4); pipe.InitBuffer(oB,R*D*4); pipe.InitBuffer(ohB,R*D*2);
        pipe.InitBuffer(mB,R*4); pipe.InitBuffer(lB,R*4); pipe.InitBuffer(alphaB,R*4);
        pipe.InitBuffer(maxAB,R*4); pipe.InitBuffer(maxBB,R*4);
        pipe.InitBuffer(sumAB,R*4); pipe.InitBuffer(sumBB,R*4);
        pipe.InitBuffer(maskB,G::CausalMaskBytes);
        pipe.InitBuffer(rowMaxBlockB,G::BroadcastBytes);
        auto score_ub=sB.template Get<float>(); auto p=pB.template Get<half>();
        auto delta=dB.template Get<float>(); auto out=oB.template Get<float>(); auto oh=ohB.template Get<half>();
        auto m=mB.template Get<float>(); auto l=lB.template Get<float>(); auto alpha=alphaB.template Get<float>();
        auto maxA=maxAB.template Get<float>(); auto maxB=maxBB.template Get<float>();
        auto sumA=sumAB.template Get<float>(); auto sumB=sumBB.template Get<float>();
        auto masks=maskB.template Get<uint32_t>();
        auto row_blocks=rowMaxBlockB.template Get<float>();
        if (group<tasks) {
            constexpr uint32_t MaskWords=G::CausalMaskBytes/sizeof(uint32_t);
            AscendC::GlobalTensor<uint32_t> masks_gm;
            masks_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(causal_masks)+sub*MaskWords,MaskWords);
            // Load each alignment once while the Cube starts the first QK.
            AscendC::DataCopy(masks,masks_gm,MaskWords);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        }
        for(uint32_t task=group;task<tasks;task+=groups) {
            const uint32_t head=task/tiles, qs=(task%tiles)*BM;
            const uint64_t base=uint64_t(head)*seq*D;
            AscendC::Duplicate(out,0.0f,R*D);
            AscendC::Duplicate(l,0.0f,R);
            AscendC::Duplicate(m,-3.402823466e+38F,R);
            AscendC::PipeBarrier<PIPE_V>();
            const int32_t first_key=(qs/BN)*BN;
            const uint32_t score_row_offset=sub*ScoreElements;
            uint32_t gm_slot=0, ub_slot=0;
            Sync<AscendC::HardEvent::V_MTE2>(pipe);
            const uint32_t mask_offset=((qs%BN)/BM)*G::CausalMaskWords;
            PrepareScoreTile<R,BN,true>(score_ub,sgm[score_row_offset],maxA,maxB,row_blocks,gm_slot,
                masks[mask_offset]);
            CompleteSoftmax<R,BN>(score_ub,p,m,l,alpha,maxA,sumA,sumB,row_blocks);
            PublishP<R,BN>(pgm,p,0,sub,pipe);
            for(int32_t ks=first_key;ks>=0;ks-=int32_t(BN),ub_slot^=1) {
                // Prior Vector work has finished with the alternate score slot and delta.
                Sync<AscendC::HardEvent::V_MTE2>(pipe);
                // Rescaling the old output only needs alpha, so queue it before the PV wait.
                AscendC::Brcb(row_blocks,alpha,R/Fp32Block,AscendC::BrcbRepeatParams(1,8));
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t col=0;col<D;col+=Fp32Repeat) {
                    AscendC::Mul(out[col],out[col],row_blocks,uint64_t{Fp32Repeat},R,
                        AscendC::BinaryRepeatParams(1,1,0,D/Fp32Block,D/Fp32Block,1));
                }
                if (ks>=int32_t(BN)) {
                    gm_slot=gm_slot+1==ScoreSlots ? 0 : gm_slot+1;
                    const uint32_t next_score_offset=gm_slot ? second_score+(gm_slot-1)*BM*BN : 0;
                    // QK for the next tile is issued before AIC waits for this P_READY.
                    const uint32_t next_slot=ub_slot^1;
                    auto next_s=score_ub[next_slot*ScoreElements];
                    auto next_p=next_slot==0 ? p : oh;
                    PrepareScoreTile<R,BN,false>(next_s,sgm[next_score_offset+score_row_offset],
                        maxA,maxB,row_blocks,gm_slot,masks);
                    CompleteSoftmax<R,BN>(next_s,next_p,m,l,alpha,maxA,sumA,sumB,row_blocks);
                    PublishP<R,BN>(pgm,next_p,next_slot,sub,pipe);
                }
                AscendC::CrossCoreWaitFlag(RelayFlag(PV_READY,ub_slot));
                AscendC::DataCopy(delta,dgm[ub_slot*BM*D+sub*R*D],R*D);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(out,out,delta,R*D);
                AscendC::CrossCoreSetFlag<0x2,PIPE_V>(RelayFlag(TILE_DONE,ub_slot));
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
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR out,GM_ADDR relay,GM_ADDR causal_masks,
    uint32_t heads,uint32_t seq,uint32_t groups)
{
    AscendC::SetSyncBaseAddr(sync_base);
    RunCore<g_coreType,BM,BN>(q,k,v,out,relay,causal_masks,heads,seq,groups);
}
#endif
} // namespace CubeFA
