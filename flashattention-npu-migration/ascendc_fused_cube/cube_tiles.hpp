// Own orchestration; CATLASS supplies documented A2 layout/copy/Mmad primitives.
#pragma once
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
    int32_t l0c_slot=0)
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
    if constexpr (DoubleBufferedL0C) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
    } else {
        Sync<AscendC::HardEvent::FIX_M>(pipe);
    }
    // TileMmad lowers directly to AscendC::Mmad (AIC Cube), FP32 accumulation.
    Catlass::Gemm::Tile::TileMmad<Arch,AT,BT,void>{}(l0c,l0a,l0b,m,n,k,true);
    if constexpr (DoubleBufferedL0C) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0c_slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0c_slot);
    } else {
        Sync<AscendC::HardEvent::M_FIX>(pipe);
    }
    typename Copies::CopyL0CToGm{}(c,l0c,Catlass::layout::RowMajor(m,n),cl0);
    if constexpr (DoubleBufferedL0C) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
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
    pipe.InitBuffer(laB,128*128*2); pipe.InitBuffer(lbB,256*128*2);
    pipe.InitBuffer(a0B,128*128*2); pipe.InitBuffer(b0B,256*128*2);
    pipe.InitBuffer(c0B,128*128*4);
    auto la=laB.Get<half>(); auto lb=lbB.Get<half>();
    auto a0=a0B.Get<half>(); auto b0=b0B.Get<half>(); auto c0=c0B.Get<float>();
    if constexpr(TransposeB) Gemm(pipe,ag,bg,cg,la,lb,a0,b0,c0,m,n,k,
        Catlass::layout::RowMajor(m,k),Catlass::layout::ColumnMajor(k,n,k));
    else Gemm(pipe,ag,bg,cg,la,lb,a0,b0,c0,m,n,k,
        Catlass::layout::RowMajor(m,k),Catlass::layout::RowMajor(k,n));
}
} // namespace CubeFA
