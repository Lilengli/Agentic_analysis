// Own mixed AIC/AIV schedule. Cube owns QK and PV, Vector owns softmax.
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
    return slot%2==0 ? QK_READY : QK_ODD_READY;
}

__aicore__ inline uint16_t RelayFlag(uint16_t flag,uint32_t slot)
{
    return slot==0 ? flag : RelayExtraBase+(slot-1)*FlagsPerRelaySlot+flag-P_READY;
}

template<bool BalanceQueries>
struct QueryAssignment {
    uint32_t groups, offset;
    uint64_t wave_base=0;

    __aicore__ inline uint64_t Task() const { return wave_base+offset; }

    __aicore__ inline void Advance()
    {
        wave_base+=groups;
        // Reverse the lanes of odd waves. The mapped task bounds the partial final wave.
        if constexpr(BalanceQueries) offset=groups-1-offset;
    }
};

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

template<bool Sum>
__aicore__ inline void ReduceFullRepeat(AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,uint32_t repeats)
{
    if constexpr(Sum) AscendC::BlockReduceSum(dst,src,repeats,64,1,1,8);
    else AscendC::BlockReduceMax(dst,src,repeats,64,1,1,8);
}

template<uint32_t Rows,bool Sum>
__aicore__ inline void ReduceRow512(AscendC::LocalTensor<float> result,
    AscendC::LocalTensor<float> scores,AscendC::LocalTensor<float> partial64,
    AscendC::LocalTensor<float> partial8)
{
    // Each block reduces eight FP32 values: 512 -> 64 -> 8 -> 1 per row.
    ReduceFullRepeat<Sum>(partial64,scores,Rows*8);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceFullRepeat<Sum>(partial8,partial64,Rows);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceFullRepeat<Sum>(result,partial8,Rows/8);
}

template<uint32_t Rows,uint32_t BN,bool ApplyCausalMask>
__aicore__ inline void TransformScores(AscendC::LocalTensor<float> s, AscendC::LocalTensor<float> maxA,
    AscendC::LocalTensor<float> maxB, AscendC::LocalTensor<float> row_blocks,
    AscendC::LocalTensor<uint32_t> causal_mask, AscendC::LocalTensor<float> partial64)
{
    AscendC::Muls(s,s,0.08838834764831845f,Rows*BN);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (ApplyCausalMask) {
        AscendC::Select(s,causal_mask,s,-3.402823466e+38F,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,Rows*BN);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // The preceding barrier has released the row-block scratch used by output rescale.
    if constexpr(BN==512) {
        ReduceRow512<Rows,false>(maxA,s,partial64,row_blocks);
    } else {
        ReduceRowMax64<Rows,BN>(maxA,s,row_blocks);
        if constexpr(BN==128) {
            AscendC::PipeBarrier<PIPE_V>();
            ReduceRowMax64<Rows,BN>(maxB,s[64],row_blocks);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(maxA,maxA,maxB,Rows);
        }
    }
}

template<uint32_t R,uint32_t BN>
__aicore__ inline void CompleteSoftmax(AscendC::LocalTensor<float> s, AscendC::LocalTensor<half> p,
    AscendC::LocalTensor<float>& m, AscendC::LocalTensor<float> l, AscendC::LocalTensor<float> alpha,
    AscendC::LocalTensor<float>& maxA, AscendC::LocalTensor<float> sumA,
    AscendC::LocalTensor<float> sumB, AscendC::LocalTensor<float> row_blocks,
    AscendC::LocalTensor<float> partial64)
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
    if constexpr(BN==512) {
        ReduceRow512<R,true>(sumA,s,partial64,row_blocks);
    } else {
        ReduceRowSum64<R,BN>(sumA,s,row_blocks);
        if constexpr(BN==128) {
            AscendC::PipeBarrier<PIPE_V>();
            ReduceRowSum64<R,BN>(sumB,s[64],row_blocks);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(sumA,sumA,sumB,R);
        }
    }
    AscendC::Cast(p,s,AscendC::RoundMode::CAST_RINT,R*BN);
    AscendC::Mul(l,l,alpha,R);
    AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(l,l,sumA,R);
}

// Keep the full-width score and P layout while transferring only the visible
// column prefix of each row. DataCopyParams strides are in 32-byte blocks.
template<uint32_t Rows,uint32_t BN,class T,class Dst,class Src>
__aicore__ inline void CopyStackPrefix(Dst dst,Src src,uint32_t active_kv)
{
    constexpr uint32_t ElementsPerBlock=32/sizeof(T);
    if(active_kv==BN) {
        AscendC::DataCopy(dst,src,Rows*BN);
    } else {
        const uint16_t blocks=active_kv/ElementsPerBlock;
        const uint16_t gap=(BN-active_kv)/ElementsPerBlock;
        AscendC::DataCopy(dst,src,AscendC::DataCopyParams(Rows,blocks,gap,gap));
    }
}

template<uint32_t Rows,uint32_t BN,bool ApplyCausalMask,bool TrimDiagonal=false>
__aicore__ inline void PrepareTile(AscendC::GlobalTensor<float> scores, AscendC::GlobalTensor<half> pgm,
    AscendC::LocalTensor<float> s, AscendC::LocalTensor<half> p, AscendC::LocalTensor<half> alternate_p,
    AscendC::LocalTensor<float> partial0, AscendC::LocalTensor<float> partial1,
    AscendC::LocalTensor<float>& m, AscendC::LocalTensor<float> l, AscendC::LocalTensor<float> alpha,
    AscendC::LocalTensor<float>& maxA, AscendC::LocalTensor<float> maxB,
    AscendC::LocalTensor<float> sumA, AscendC::LocalTensor<float> sumB,
    AscendC::LocalTensor<float> row_blocks, AscendC::LocalTensor<uint32_t> mask,
    uint32_t gm_slot,uint32_t pv_slot,uint32_t sub,AscendC::TPipe& pipe,uint32_t active_kv=BN)
{
    AscendC::CrossCoreWaitFlag(ScoreReadyFlag(gm_slot));
    if constexpr(BN!=512) {
        AscendC::DataCopy(s,scores,Rows*BN);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        TransformScores<Rows,BN,ApplyCausalMask>(s,maxA,maxB,row_blocks,mask,row_blocks);
        CompleteSoftmax<Rows,BN>(s,p,m,l,alpha,maxA,sumA,sumB,row_blocks,row_blocks);
        PublishP<Rows,BN>(pgm,p,pv_slot,sub,pipe);
    } else {
        using G=Geometry<2*Rows,BN>;
        constexpr uint32_t BatchRows=G::BatchRows, Elements=G::BatchElements;
        static_assert(Rows==4*BatchRows);
        if constexpr(TrimDiagonal) {
            static_assert(ApplyCausalMask);
            // Unwritten columns must be initialized before vector arithmetic.
            // The softmax mask keeps these lanes at zero after each reuse.
            if(active_kv<BN) AscendC::Duplicate(s,0.0f,UbScoreSlots*Elements);
        }
        Sync<AscendC::HardEvent::V_MTE2>(pipe);
        if constexpr(TrimDiagonal) CopyStackPrefix<BatchRows,BN,float>(s,scores,active_kv);
        else AscendC::DataCopy(s,scores,Elements);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        for(uint32_t batch=0;batch<Rows/BatchRows;++batch) {
            const uint32_t slot=batch%2, row=batch*BatchRows;
            const int32_t score_event=slot==0 ? 0 : 3;
            if(batch+1<Rows/BatchRows) {
                if(batch>0) Sync<AscendC::HardEvent::V_MTE2>(pipe);
                if constexpr(TrimDiagonal)
                    CopyStackPrefix<BatchRows,BN,float>(s[(slot^1)*Elements],
                        scores[(batch+1)*Elements],active_kv);
                else
                    AscendC::DataCopy(s[(slot^1)*Elements],scores[(batch+1)*Elements],Elements);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot==0 ? 3 : 0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(score_event);
            // Both the FP32 reduction scratch and FP16 P reuse this batch's output buffer.
            if(batch>=2) AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
            auto scores_ub=s[slot*Elements];
            auto batch_p=slot==0 ? p : alternate_p;
            auto partial=slot==0 ? partial0 : partial1;
            auto batch_m=m[row], batch_max=maxA[row];
            TransformScores<BatchRows,BN,ApplyCausalMask>(scores_ub,batch_max,maxB,row_blocks,
                mask[batch*Elements/32],partial);
            CompleteSoftmax<BatchRows,BN>(scores_ub,batch_p,batch_m,l[row],alpha[row],
                batch_max,sumA,sumB,row_blocks,partial);
            Sync<AscendC::HardEvent::V_MTE3>(pipe);
            auto pdst=pgm[(pv_slot*2+sub)*Rows*BN+batch*Elements];
            if constexpr(TrimDiagonal) CopyStackPrefix<BatchRows,BN,half>(pdst,batch_p,active_kv);
            else AscendC::DataCopy(pdst,batch_p,Elements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
        // Each row slice wrote to the alternate running-max buffer.
        auto previous_m=m;
        m=maxA;
        maxA=previous_m;
        AscendC::CrossCoreSetFlag<0x2,PIPE_MTE3>(RelayFlag(P_READY,pv_slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
    }
}

template<int Core,uint32_t BM,uint32_t BN,bool PrefetchV=false,bool CrossQuery=false,
    bool TrimDiagonal=false,bool VectorNormalize=false,bool BalanceQueries=false>
__aicore__ inline void RunCore(GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR output,
    GM_ADDR relay,GM_ADDR causal_masks,uint32_t heads,uint32_t seq,uint32_t groups)
{
    using G=Geometry<BM,BN>;
    static_assert(!PrefetchV || (BM==128 && BN==512));
    static_assert(!CrossQuery || PrefetchV);
    static_assert(!TrimDiagonal || (BM==128 && BN==512 && CrossQuery));
    static_assert(!VectorNormalize || TrimDiagonal);
    static_assert(!BalanceQueries || TrimDiagonal);
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
        constexpr uint32_t CubeKv=G::CubeKv;
        constexpr uint32_t MaxK=CubeKv>D?CubeKv:D, MaxN=MaxK;
        constexpr uint32_t L0CSlotElements=BM*MaxN;
        static_assert(2*L0CSlotElements*sizeof(float)<=128*1024);
        AscendC::TBuf<AscendC::TPosition::A1> qB,pB,bB,pvVB;
        AscendC::TBuf<AscendC::TPosition::A2> a0B;
        AscendC::TBuf<AscendC::TPosition::B2> b0B;
        AscendC::TBuf<AscendC::TPosition::CO1> c0B;
        pipe.InitBuffer(qB,BM*D*sizeof(half));
        pipe.InitBuffer(pB,BM*CubeKv*sizeof(half));
        pipe.InitBuffer(bB,CubeKv*D*sizeof(half));
        if constexpr(PrefetchV) {
            static_assert((BM*D+BM*CubeKv+CubeKv*D+BN*D)*sizeof(half)<=512*1024);
            pipe.InitBuffer(pvVB,BN*D*sizeof(half));
        }
        pipe.InitBuffer(a0B,BM*MaxK*sizeof(half));
        pipe.InitBuffer(b0B,CubeKv*D*sizeof(half));
        pipe.InitBuffer(c0B,2*L0CSlotElements*sizeof(float));
        auto ql=qB.template Get<half>(); auto pl=pB.template Get<half>(); auto bl=bB.template Get<half>();
        auto a0=a0B.template Get<half>(); auto b0=b0B.template Get<half>(); auto c0=c0B.template Get<float>();
        // Reserve FIX_M and M_FIX event IDs 0 and 1 for these two L0C slots.
        // Each FIX return releases only the slot its copy read.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        int32_t accum_slot=0;
        // Cross-query mode carries ownership of each PV relay slot across tasks.
        uint32_t pending_pv=0;
        for(QueryAssignment<BalanceQueries> assignment{groups,group};assignment.Task()<tasks;assignment.Advance()) {
            const uint32_t task=static_cast<uint32_t>(assignment.Task());
            const uint32_t head=task/tiles, qs=(task%tiles)*BM;
            const uint64_t base=uint64_t(head)*seq*D;
            constexpr int32_t step=BN;
            const int32_t first_key=(qs/BN)*BN;
            const uint32_t diagonal_kv=TrimDiagonal ? qs%BN+BM : BN;
            uint32_t slot=0, issued_pv=0;
            for (int32_t qk_key=first_key, pv_key=first_key+int32_t(QkLookaheadTiles)*step;
                 pv_key>=0; qk_key-=step, pv_key-=step, slot=slot+1==ScoreSlots ? 0 : slot+1) {
                if (qk_key>=0) {
                    const uint32_t score_offset=slot ? second_score+(slot-1)*BM*BN : 0;
                    const uint32_t qk_kv=qk_key==first_key ? diagonal_kv : BN;
                    for(uint32_t kv=0;kv<qk_kv;kv+=CubeKv) {
                        Gemm<true>(pipe,qgm[base+uint64_t(qs)*D],kgm[base+(uint64_t(qk_key)+kv)*D],
                            sgm[score_offset+kv],ql,bl,a0,b0,c0[accum_slot*L0CSlotElements],
                            BM,CubeKv,D,Catlass::layout::RowMajor(BM,D),
                            Catlass::layout::ColumnMajor(D,CubeKv,D),qk_key==first_key && kv==0,
                            accum_slot,BN);
                        accum_slot^=1;
                    }
                    AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(ScoreReadyFlag(slot));
                }
                // Prime the score queue before the first PV.
                if (pv_key>first_key) continue;
                const uint32_t pv_slot=issued_pv%PvRelaySlots;
                const uint32_t pv_kv=pv_key==first_key ? diagonal_kv : BN;
                auto pv_bl=bl;
                if constexpr(PrefetchV) {
                    pv_bl=pvVB.template Get<half>();
                    PrefetchPvV<CubeKv,D,BN>(pipe,vgm[base+uint64_t(pv_key)*D],pv_bl,pv_kv);
                }
                if constexpr(CrossQuery) {
                    if(pending_pv & (1u<<pv_slot))
                        AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,pv_slot));
                } else {
                    if(issued_pv>=PvRelaySlots) AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,pv_slot));
                }
                AscendC::CrossCoreWaitFlag(RelayFlag(P_READY,pv_slot));
                // Accumulate all KV slices in the same FP32 L0C slot before Fixpipe.
                for(uint32_t kv=0;kv<pv_kv;kv+=CubeKv) {
                    auto pv_slice=pv_bl;
                    if constexpr(PrefetchV) pv_slice=pv_bl[kv*D];
                    Gemm<true>(pipe,pgm[pv_slot*BM*BN+kv],vgm[base+(uint64_t(pv_key)+kv)*D],
                        dgm[pv_slot*BM*D],pl,pv_slice,a0,b0,c0[accum_slot*L0CSlotElements],
                        BM,D,CubeKv,Catlass::layout::RowMajor(BM,CubeKv,BN),
                        Catlass::layout::RowMajor(CubeKv,D),true,accum_slot,D,kv==0,kv+CubeKv==pv_kv,!PrefetchV);
                }
                accum_slot^=1;
                AscendC::CrossCoreSetFlag<0x2,PIPE_FIX>(RelayFlag(PV_READY,pv_slot));
                ++issued_pv;
                if constexpr(CrossQuery) pending_pv|=1u<<pv_slot;
            }
            if constexpr(!CrossQuery) {
                const uint32_t first_pending=issued_pv>PvRelaySlots ? issued_pv-PvRelaySlots : 0;
                for(uint32_t tile=first_pending;tile<issued_pv;++tile)
                    AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,tile%PvRelaySlots));
            }
            // All P waits are consumed, so both AIVs have read the prior scores.
            // The next query's QK can safely reuse the score slots while its
            // predecessor's last PVs are still being consumed.
        }
        if constexpr(CrossQuery) {
            for(uint32_t slot=0;slot<PvRelaySlots;++slot)
                if(pending_pv & (1u<<slot)) AscendC::CrossCoreWaitFlag(RelayFlag(TILE_DONE,slot));
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    } else {
        constexpr uint32_t R=G::Rows;
        constexpr uint32_t ScoreElements=R*BN;
        constexpr uint32_t Fp32Block=32/sizeof(float), Fp32Repeat=256/sizeof(float);
        static_assert(R%Fp32Block==0 && BN%Fp32Repeat==0 && D%Fp32Repeat==0);
        static_assert(G::BatchElements<=R*D);  // The final-output staging buffer also holds a P batch.
        static_assert(G::UbBytes+SelectScratchBytes<=192*1024-256);
        AscendC::TBuf<AscendC::TPosition::VECCALC> sB,pB,dB,oB,ohB,mB,lB,alphaB,maxAB,maxBB,sumAB,sumBB,maskB;
        AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxBlockB;
        pipe.InitBuffer(sB,UbScoreSlots*G::BatchElements*sizeof(float));
        pipe.InitBuffer(pB,G::PBufferSlots*G::BatchElements*sizeof(half));
        pipe.InitBuffer(dB,R*D*4); pipe.InitBuffer(oB,R*D*4); pipe.InitBuffer(ohB,R*D*2);
        pipe.InitBuffer(mB,R*4); pipe.InitBuffer(lB,R*4); pipe.InitBuffer(alphaB,PvRelaySlots*R*4);
        pipe.InitBuffer(maxAB,R*4); pipe.InitBuffer(maxBB,R*4);
        pipe.InitBuffer(sumAB,R*4); pipe.InitBuffer(sumBB,R*4);
        pipe.InitBuffer(maskB,G::CausalMaskBytes);
        pipe.InitBuffer(rowMaxBlockB,G::BroadcastBytes);
        auto score_ub=sB.template Get<float>(); auto p=pB.template Get<half>();
        auto delta=dB.template Get<float>(); auto out=oB.template Get<float>(); auto oh=ohB.template Get<half>();
        auto partial0=pB.template Get<float>(); auto partial1=ohB.template Get<float>();
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
        for(QueryAssignment<BalanceQueries> assignment{groups,group};assignment.Task()<tasks;assignment.Advance()) {
            const uint32_t task=static_cast<uint32_t>(assignment.Task());
            const uint32_t head=task/tiles, qs=(task%tiles)*BM;
            const uint64_t base=uint64_t(head)*seq*D;
            AscendC::Duplicate(out,0.0f,R*D);
            AscendC::Duplicate(l,0.0f,R);
            AscendC::Duplicate(m,-3.402823466e+38F,R);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t kv_tiles=qs/BN+1;
            const uint32_t score_row_offset=sub*ScoreElements;
            Sync<AscendC::HardEvent::V_MTE2>(pipe);
            const uint32_t mask_offset=((qs%BN)/BM)*G::CausalMaskWords;
            const uint32_t diagonal_kv=TrimDiagonal ? qs%BN+BM : BN;
            PrepareTile<R,BN,true,TrimDiagonal>(sgm[score_row_offset],pgm,score_ub,p,oh,partial0,partial1,
                m,l,alpha,maxA,maxB,sumA,sumB,row_blocks,masks[mask_offset],0,0,sub,pipe,diagonal_kv);
            uint32_t prepared=1;
            for(uint32_t tile=0;tile<kv_tiles;++tile) {
                const uint32_t pv_slot=tile%PvRelaySlots;
                // Prior Vector work has finished with the score slots and delta.
                Sync<AscendC::HardEvent::V_MTE2>(pipe);
                // Rescaling the old output only needs alpha, so queue it before the PV wait.
                AscendC::Brcb(row_blocks,alpha[pv_slot*R],R/Fp32Block,AscendC::BrcbRepeatParams(1,8));
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t col=0;col<D;col+=Fp32Repeat) {
                    AscendC::Mul(out[col],out[col],row_blocks,uint64_t{Fp32Repeat},R,
                        AscendC::BinaryRepeatParams(1,1,0,D/Fp32Block,D/Fp32Block,1));
                }
                for(;prepared<kv_tiles && prepared<tile+PvRelaySlots;++prepared) {
                    const uint32_t gm_slot=prepared%ScoreSlots;
                    const uint32_t next_score_offset=gm_slot ? second_score+(gm_slot-1)*BM*BN : 0;
                    const uint32_t next_slot=prepared%PvRelaySlots;
                    auto next_s=score_ub;
                    auto next_p=p;
                    if constexpr(BN!=512) {
                        next_s=score_ub[(prepared%UbScoreSlots)*ScoreElements];
                        next_p=next_slot==PvRelaySlots-1 ? oh : p[next_slot*ScoreElements];
                    }
                    // The previous PV from this slot was consumed before reusing its P and alpha.
                    PrepareTile<R,BN,false>(sgm[next_score_offset+score_row_offset],pgm,next_s,next_p,
                        oh,partial0,partial1,m,l,alpha[next_slot*R],maxA,maxB,sumA,sumB,row_blocks,
                        masks,gm_slot,next_slot,sub,pipe);
                }
                AscendC::CrossCoreWaitFlag(RelayFlag(PV_READY,pv_slot));
                AscendC::DataCopy(delta,dgm[pv_slot*BM*D+sub*R*D],R*D);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::PipeBarrier<PIPE_V>(); AscendC::Add(out,out,delta,R*D);
                AscendC::CrossCoreSetFlag<0x2,PIPE_V>(RelayFlag(TILE_DONE,pv_slot));
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr(VectorNormalize) {
                // Each Div repeat normalizes 64 columns of one row using its broadcast denominator.
                AscendC::Brcb(row_blocks,l,R/Fp32Block,AscendC::BrcbRepeatParams(1,8));
                AscendC::PipeBarrier<PIPE_V>();
                for(uint32_t col=0;col<D;col+=Fp32Repeat) {
                    AscendC::Div(out[col],out[col],row_blocks,uint64_t{Fp32Repeat},R,
                        AscendC::BinaryRepeatParams(1,1,0,D/Fp32Block,D/Fp32Block,1));
                }
            } else {
                Sync<AscendC::HardEvent::V_S>(pipe);
                for(uint32_t r=0;r<R;++r) AscendC::Muls(out[r*D],out[r*D],1.0f/l.GetValue(r),D);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(oh,out,AscendC::RoundMode::CAST_RINT,R*D);
            Sync<AscendC::HardEvent::V_MTE3>(pipe);
            AscendC::DataCopy(ogm[base+uint64_t(qs+sub*R)*D],oh,R*D);
            Sync<AscendC::HardEvent::MTE3_V>(pipe);
        }
    }
}

#ifndef CUBE_FA_CPU_SIM
template<uint32_t BM,uint32_t BN,bool PrefetchV=false,bool CrossQuery=false,
    bool TrimDiagonal=false,bool VectorNormalize=false,bool BalanceQueries=false>
__global__ __mix__(1,2) void fused_cube_attention(uint64_t sync_base,
    GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR out,GM_ADDR relay,GM_ADDR causal_masks,
    uint32_t heads,uint32_t seq,uint32_t groups)
{
    AscendC::SetSyncBaseAddr(sync_base);
    RunCore<g_coreType,BM,BN,PrefetchV,CrossQuery,TrimDiagonal,VectorNormalize,BalanceQueries>(
        q,k,v,out,relay,causal_masks,heads,seq,groups);
}
#endif
} // namespace CubeFA
