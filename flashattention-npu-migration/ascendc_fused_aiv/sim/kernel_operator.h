// CPU semantic mock of only the Ascend C APIs used by fused_kernel.hpp.
// It checks indexing/math, not device compilation, ordering or hardware behavior.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#define __aicore__
#define __global__
#define __vector__
#define __gm__
// Portable binary16 for older host compilers; device code uses CANN's half.
struct half {
    uint16_t bits=0;
    half()=default;
    half(float f) {
        uint32_t u; std::memcpy(&u,&f,4);
        uint16_t sign=(u>>16)&0x8000;
        int e=int((u>>23)&255)-127+15;
        uint32_t m=u&0x7fffff;
        if(((u>>23)&255)==255) { bits=sign|0x7c00|(m?0x0200:0); return; }
        if(e>=31) { bits=sign|0x7c00; return; }
        if(e<=0) {
            if(e < -10) { bits=sign; return; }
            m|=0x800000;
            unsigned shift=14-e;
            uint32_t value=m>>shift, rem=m&((1u<<shift)-1), midpoint=1u<<(shift-1);
            bits=sign|(value+(rem>midpoint || (rem==midpoint && (value&1)))); return;
        }
        m+=0xfff+((m>>13)&1);
        bits=sign|uint16_t((e<<10)+(m>>13));
    }
    operator float() const {
        unsigned e=(bits>>10)&31, m=bits&1023;
        float value=e==31 ? (m?NAN:INFINITY) : e==0 ? std::ldexp(float(m),-24) : std::ldexp(float(1024+m),int(e)-25);
        return (bits&0x8000)?-value:value;
    }
    bool operator==(half other) const { return bits==other.bits; }
    bool operator!=(half other) const { return !(*this==other); }
};
static_assert(sizeof(half)==2);
using GM_ADDR = uint8_t*;
constexpr int PIPE_V = 0;

namespace AscendC {
inline uint32_t block_idx = 0, block_num = 1;
inline uint64_t gm_read_elements = 0, gm_write_elements = 0;
inline uint32_t GetBlockIdx() { return block_idx; }
inline uint32_t GetBlockNum() { return block_num; }
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class HardEvent { V_MTE2, MTE2_V, V_S, S_V, V_MTE3, MTE3_S };
template <HardEvent E> void SetFlag(int) {}
template <HardEvent E> void WaitFlag(int) {}
template <int P> void PipeBarrier() {}

template <class T> struct LocalTensor {
    std::shared_ptr<std::vector<uint8_t>> owner;
    size_t offset = 0, length = 0;
    T GetValue(size_t i) const { assert(i < length); T x; std::memcpy(&x, owner->data()+offset+i*sizeof(T), sizeof(T)); return x; }
    void SetValue(size_t i, T x) const { assert(i < length); std::memcpy(owner->data()+offset+i*sizeof(T), &x, sizeof(T)); }
    LocalTensor operator[](size_t i) const { assert(i <= length); return {owner, offset+i*sizeof(T), length-i}; }
};
template <class T> struct GlobalTensor {
    T* data = nullptr;
    size_t length = 0, offset = 0;
    void SetGlobalBuffer(T* p, size_t n) { data=p; length=n; }
    GlobalTensor operator[](size_t i) const { assert(i <= length); return {data+i, length-i, offset+i*sizeof(T)}; }
    // Deliberately no GetValue/SetValue: all device GM traffic must use DMA.
};
template <TPosition P> struct TBuf {
    std::shared_ptr<std::vector<uint8_t>> storage;
    template <class T> LocalTensor<T> Get() { return {storage, 0, storage->size()/sizeof(T)}; }
};
struct TPipe {
    template <TPosition P> void InitBuffer(TBuf<P>& b, size_t bytes) {
        assert(bytes % 32 == 0);
        b.storage = std::make_shared<std::vector<uint8_t>>(bytes, 0xff);
    }
    int FetchEventID(HardEvent) { return 0; }
};
template <class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, size_t n) {
    assert(dst.offset%32==0 && src.offset%32==0 && n*sizeof(T)%32==0);
    assert(n<=dst.length && n<=src.length);
    for(size_t i=0;i<n;++i) dst.SetValue(i,src.data[i]);
    gm_read_elements += n;
}
template <class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t n) {
    assert(dst.offset%32==0 && src.offset%32==0 && n*sizeof(T)%32==0);
    assert(n<=dst.length && n<=src.length);
    for(size_t i=0;i<n;++i) dst.data[i]=src.GetValue(i);
    gm_write_elements += n;
}
template <class A, class B> void Cast(LocalTensor<A> dst, LocalTensor<B> src, RoundMode, size_t n) {
    for(size_t i=0;i<n;++i) dst.SetValue(i,static_cast<A>(src.GetValue(i)));
}
template <class T> void Duplicate(LocalTensor<T> dst, T v, size_t n) {
    for(size_t i=0;i<n;++i) dst.SetValue(i,v);
}
struct BinaryRepeatParams { int dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride; };
template <class T> void Mul(LocalTensor<T> dst, LocalTensor<T> x, LocalTensor<T> y,
                            int mask, int repeats, BinaryRepeatParams p) {
    constexpr int block = 32/sizeof(T);
    for(int r=0;r<repeats;++r) for(int i=0;i<mask;++i) {
        auto index=[&](int rep,int blk) { return r*rep*block+(i/block)*blk*block+i%block; };
        dst.SetValue(index(p.dstRepStride,p.dstBlkStride),
            x.GetValue(index(p.src0RepStride,p.src0BlkStride))*y.GetValue(index(p.src1RepStride,p.src1BlkStride)));
    }
}
template <class T> void WholeReduceSum(LocalTensor<T> dst, LocalTensor<T> src,
                                      int mask,int repeats,int dstStride,int blockStride,int repStride) {
    constexpr int block=32/sizeof(T);
    for(int r=0;r<repeats;++r) {
        std::vector<T> values(mask);
        for(int i=0;i<mask;++i) values[i]=src.GetValue(r*repStride*block+(i/block)*blockStride*block+i%block);
        for(int n=mask;n>1;n/=2) for(int i=0;i<n/2;++i) values[i]=values[2*i]+values[2*i+1];
        dst.SetValue(r*dstStride,values[0]);
    }
}
template <class T> void Add(LocalTensor<T> dst, LocalTensor<T> x, LocalTensor<T> y, size_t n) {
    for(size_t i=0;i<n;++i) dst.SetValue(i,x.GetValue(i)+y.GetValue(i));
}
template <class T> void Muls(LocalTensor<T> dst, LocalTensor<T> x,T scale,size_t n) {
    for(size_t i=0;i<n;++i) dst.SetValue(i,x.GetValue(i)*scale);
}
template <class T> void Exp(LocalTensor<T> dst,LocalTensor<T> x,size_t n) {
    for(size_t i=0;i<n;++i) dst.SetValue(i,std::exp(x.GetValue(i)));
}
struct BrcbRepeatParams { int dstBlkStride, dstRepStride; };
template <class T> void Brcb(LocalTensor<T> dst,LocalTensor<T> x,int repeats,BrcbRepeatParams p) {
    constexpr int block=32/sizeof(T);
    for(int r=0;r<repeats;++r) for(int j=0;j<8;++j) for(int i=0;i<block;++i)
        dst.SetValue((r*p.dstRepStride+j*p.dstBlkStride)*block+i,x.GetValue(r*8+j));
}
} // namespace AscendC
