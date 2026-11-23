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

// Independent V staging: issue the same packed L1 copies as the PV loop
// without waiting for P (or for MTE2 to finish). The PV GEMM's MTE2_MTE1
// dependency covers the prefetched V as well as the subsequent P copy.
template<uint32_t SliceKv,uint32_t HeadDim,uint32_t StackKv>
__aicore__ inline void PrefetchPvV(AscendC::TPipe& pipe,AscendC::GlobalTensor<half> v,
    AscendC::LocalTensor<half> staged,uint32_t active_kv=StackKv)
{
    using B=Catlass::Gemm::GemmType<half,Catlass::layout::RowMajor>;
    using Copy=Catlass::Gemm::Tile::CopyGmToL1<Catlass::Arch::AtlasA2,B>;
    using Layout=typename Copy::LayoutDst;
    auto layout=Layout::template MakeLayout<half>(SliceKv,HeadDim);
    // Prior PV may still be reading this staging area on MTE1.
    Sync<AscendC::HardEvent::MTE1_MTE2>(pipe);
    for(uint32_t kv=0;kv<active_kv;kv+=SliceKv)
        Copy{}(staged[kv*HeadDim],v[kv*HeadDim],layout,
            Catlass::layout::RowMajor(SliceKv,HeadDim));
}

template<bool DoubleBufferedL0C=false, bool ReuseL0A=false, class ALayout, class BLayout>
__aicore__ inline void Gemm(AscendC::TPipe& pipe,
    AscendC::GlobalTensor<half> a, AscendC::GlobalTensor<half> b,
    AscendC::GlobalTensor<float> c, AscendC::LocalTensor<half> la,
    AscendC::LocalTensor<half> lb, AscendC::LocalTensor<half> l0a,
    AscendC::LocalTensor<half> l0b, AscendC::LocalTensor<float> l0c,
    uint32_t m,uint32_t n,uint32_t k,ALayout a_layout,BLayout b_layout,bool load_a=true,
    int32_t l0c_slot=0,uint32_t c_stride=0,bool init_c=true,bool store_c=true,bool load_b=true)
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
    if(load_b) typename Copies::CopyGmToL1B{}(lb,b,bl1,b_layout);
    Sync<AscendC::HardEvent::MTE2_MTE1>(pipe);
    Sync<AscendC::HardEvent::M_MTE1>(pipe);
    // In the resident Q path the first GM load also establishes the L0A copy.
    if (!ReuseL0A || load_a) typename Copies::CopyL1ToL0A{}(l0a,la,al0,al1);
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
