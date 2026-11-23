// Own orchestration; CATLASS supplies documented A2 layout/copy/Mmad primitives.
#pragma once
#include "geometry.hpp"
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"

namespace CubeFA {
template <AscendC::HardEvent E>
__aicore__ inline void Sync(AscendC::TPipe& pipe) {
    auto id=static_cast<int32_t>(pipe.FetchEventID(E));
    AscendC::SetFlag<E>(id); AscendC::WaitFlag<E>(id);
}

template<bool DoubleBufferedL0C=false, class ALayout, class BLayout>
__aicore__ inline void Gemm(AscendC::TPipe& pipe,
    AscendC::GlobalTensor<half> a, AscendC::GlobalTensor<half> b,
    AscendC::GlobalTensor<float> c, AscendC::LocalTensor<half> la,
    AscendC::LocalTensor<half> lb, AscendC::LocalTensor<half> l0a,
    AscendC::LocalTensor<half> l0b, AscendC::LocalTensor<float> l0c,
    uint32_t m,uint32_t n,uint32_t k,ALayout a_layout,BLayout b_layout,bool load_a=true,
    int32_t l0c_slot=0,uint32_t c_stride=0,bool init_c=true,bool store_c=true)
{
    using Arch=Catlass::Arch::AtlasA2;
    using AT=Catlass::Gemm::GemmType<half,ALayout>;
    using BT=Catlass::Gemm::GemmType<half,BLayout>;
    using CT=Catlass::Gemm::GemmType<float,Catlass::layout::RowMajor>;
    using Copies=Catlass::Gemm::Tile::TileCopy<Arch,AT,BT,CT>;
    using AL1=typename Copies::CopyGmToL1A::LayoutDst;
    using BL1=typename Copies::CopyGmToL1B::LayoutDst;
    using AL0=typename Copies::CopyL1ToL0A::LayoutDst;
    using BL0=typename Copies::CopyL1ToL0B::LayoutDst;
    auto al1=AL1::template MakeLayout<half>(m,k);
    auto bl1=BL1::template MakeLayout<half>(k,n);
    auto al0=AL0::template MakeLayout<half>(m,k);
    auto bl0=BL0::template MakeLayout<half>(k,n);
    auto cl0=Catlass::layout::zN::MakeLayoutInL0C(Catlass::MakeCoord(m,n));
    Sync<AscendC::HardEvent::MTE1_MTE2>(pipe);
    if(load_a) typename Copies::CopyGmToL1A{}(la,a,al1,a_layout);
    typename Copies::CopyGmToL1B{}(lb,b,bl1,b_layout);
    Sync<AscendC::HardEvent::MTE2_MTE1>(pipe);
    Sync<AscendC::HardEvent::M_MTE1>(pipe);
    typename Copies::CopyL1ToL0A{}(l0a,la,al0,al1);
    typename Copies::CopyL1ToL0B{}(l0b,lb,bl0,bl1);
    Sync<AscendC::HardEvent::MTE1_M>(pipe);
    if (init_c) {
        if constexpr (DoubleBufferedL0C) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
        } else {
            Sync<AscendC::HardEvent::FIX_M>(pipe);
        }
    }
    // TileMmad lowers directly to AscendC::Mmad (AIC Cube), FP32 accumulation.
    Catlass::Gemm::Tile::TileMmad<Arch,AT,BT,void>{}(l0c,l0a,l0b,m,n,k,init_c);
    if (store_c) {
        if constexpr (DoubleBufferedL0C) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0c_slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0c_slot);
        } else {
            Sync<AscendC::HardEvent::M_FIX>(pipe);
        }
        const uint32_t stride = c_stride ? c_stride : n;
        typename Copies::CopyL0CToGm{}(c,l0c,Catlass::layout::RowMajor(m,n,stride),cl0);
        if constexpr (DoubleBufferedL0C) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
        }
    }
}

// One 32-row tranche from each AIV. P stays in L0A for all four
// feature tiles. Each B tile serves both AIV row blocks before its slot is reused.
template<bool LoadV>
__aicore__ inline void PvFeatureStripe(AscendC::TPipe& pipe,
    AscendC::GlobalTensor<half> p, AscendC::GlobalTensor<half> v,
    AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<half> l1p,
    AscendC::LocalTensor<half> l1v, AscendC::LocalTensor<half> l0a,
    AscendC::LocalTensor<half> l0b, AscendC::LocalTensor<float> l0c,
    uint32_t l0c_slot_elements)
{
    constexpr uint32_t M=PvRowTile, N=PvFeatureTile, K=PvFullKv;
    constexpr uint32_t PElements=M*K, VElements=K*N;
    constexpr uint32_t Features=D/N;
    static_assert(Features==4 && PvBSlots==2);
    using Arch=Catlass::Arch::AtlasA2;
    using HalfRM=Catlass::Gemm::GemmType<half,Catlass::layout::RowMajor>;
    using FloatRM=Catlass::Gemm::GemmType<float,Catlass::layout::RowMajor>;
    using Copies=Catlass::Gemm::Tile::TileCopy<Arch,HalfRM,HalfRM,FloatRM>;
    using AL1=typename Copies::CopyGmToL1A::LayoutDst;
    using BL1=typename Copies::CopyGmToL1B::LayoutDst;
    using AL0=typename Copies::CopyL1ToL0A::LayoutDst;
    using BL0=typename Copies::CopyL1ToL0B::LayoutDst;
    auto al1=AL1::template MakeLayout<half>(M,K);
    auto bl1=BL1::template MakeLayout<half>(K,N);
    auto al0=AL0::template MakeLayout<half>(M,K);
    auto bl0=BL0::template MakeLayout<half>(K,N);
    auto cl0=Catlass::layout::zN::MakeLayoutInL0C(Catlass::MakeCoord(M,N));
    const uint32_t aiv_row_stride=2*M;

    // The QK path and consecutive PV calls share these L1 and L0 allocations.
    Sync<AscendC::HardEvent::MTE1_MTE2>(pipe);
    for(uint32_t sub=0;sub<2;++sub)
        typename Copies::CopyGmToL1A{}(l1p[sub*PElements],p[sub*aiv_row_stride*K],
            al1,Catlass::layout::RowMajor(M,K));
    Sync<AscendC::HardEvent::MTE2_MTE1>(pipe);
    Sync<AscendC::HardEvent::M_MTE1>(pipe);
    for(uint32_t sub=0;sub<2;++sub)
        typename Copies::CopyL1ToL0A{}(l0a[sub*PElements],l1p[sub*PElements],al0,al1);

    const int32_t ready[2]={
        static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE1_M>()),
        static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::MTE1_M>())};
    const int32_t released[2]={
        static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::M_MTE1>()),
        static_cast<int32_t>(pipe.AllocEventID<AscendC::HardEvent::M_MTE1>())};
    for(uint32_t feature=0;feature<Features;++feature) {
        const uint32_t col=feature*N, slot=feature%PvBSlots;
        if constexpr(LoadV) {
            typename Copies::CopyGmToL1B{}(l1v[feature*VElements],v[col],bl1,
                Catlass::layout::RowMajor(K,N,D));
            Sync<AscendC::HardEvent::MTE2_MTE1>(pipe);
        }
        if(feature>=PvBSlots)
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(released[slot]);
        auto b=l0b[slot*VElements];
        typename Copies::CopyL1ToL0B{}(b,l1v[feature*VElements],bl0,bl1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ready[slot]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ready[slot]);
        for(uint32_t sub=0;sub<2;++sub) {
            auto c=l0c[sub*l0c_slot_elements+feature*M*N];
            if(feature==0) AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(sub);
            Catlass::Gemm::Tile::TileMmad<Arch,HalfRM,HalfRM,void>{}(
                c,l0a[sub*PElements],b,M,N,K);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(sub);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(sub);
            typename Copies::CopyL0CToGm{}(dst[sub*aiv_row_stride*D+col],c,
                Catlass::layout::RowMajor(M,N,D),cl0);
            if(feature+1==Features) AscendC::SetFlag<AscendC::HardEvent::FIX_M>(sub);
        }
        if(feature+PvBSlots<Features)
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(released[slot]);
    }
    for(uint32_t slot=0;slot<PvBSlots;++slot) {
        pipe.ReleaseEventID<AscendC::HardEvent::MTE1_M>(ready[slot]);
        pipe.ReleaseEventID<AscendC::HardEvent::M_MTE1>(released[slot]);
    }
}

// Isolate physical Cube/layout behavior from cross-core scheduling in tests.
template<bool TransposeB>
__global__ __cube__ void cube_product_probe(GM_ADDR a,GM_ADDR b,GM_ADDR c,
    uint32_t m,uint32_t n,uint32_t k)
{
    AscendC::GlobalTensor<half> ag,bg;
    AscendC::GlobalTensor<float> cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a),m*k);
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b),k*n);
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c),m*n);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> laB,lbB;
    AscendC::TBuf<AscendC::TPosition::A2> a0B;
    AscendC::TBuf<AscendC::TPosition::B2> b0B;
    AscendC::TBuf<AscendC::TPosition::CO1> c0B;
    pipe.InitBuffer(laB,128*128*2); pipe.InitBuffer(lbB,128*128*2);
    pipe.InitBuffer(a0B,128*128*2); pipe.InitBuffer(b0B,128*128*2);
    pipe.InitBuffer(c0B,128*128*4);
    auto la=laB.Get<half>(); auto lb=lbB.Get<half>();
    auto a0=a0B.Get<half>(); auto b0=b0B.Get<half>(); auto c0=c0B.Get<float>();
    if constexpr(TransposeB) Gemm(pipe,ag,bg,cg,la,lb,a0,b0,c0,m,n,k,
        Catlass::layout::RowMajor(m,k),Catlass::layout::ColumnMajor(k,n,k));
    else Gemm(pipe,ag,bg,cg,la,lb,a0,b0,c0,m,n,k,
        Catlass::layout::RowMajor(m,k),Catlass::layout::RowMajor(k,n));
}
} // namespace CubeFA
