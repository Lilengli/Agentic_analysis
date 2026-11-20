#pragma once
#include <cstdint>

namespace CubeFA {
constexpr uint32_t D=128;
constexpr uint32_t ScoreSlots=8;
constexpr uint32_t QkLookaheadTiles=ScoreSlots-1;
constexpr uint32_t UbScoreSlots=2;
constexpr uint32_t SelectScratchBytes=8*1024;
constexpr uint32_t RelayBytesFor(uint32_t bm, uint32_t bn)
{
    return ScoreSlots * bm * bn * sizeof(float) + bm * bn * sizeof(uint16_t) + bm * D * sizeof(float);
}
constexpr uint32_t MaskWordsPerAiv(uint32_t bn)
{
    return (bn/2)*(bn/32);
}

inline void PackCausalMasks(uint32_t* masks, uint32_t bm, uint32_t bn)
{
    const uint32_t rows=bm/2, words_per_row=bn/32;
    const uint32_t words_per_aiv=MaskWordsPerAiv(bn);
    for (uint32_t sub=0;sub<2;++sub) {
        for (uint32_t stripe=0;stripe<bn/bm;++stripe) {
            for (uint32_t row=0;row<rows;++row) {
                const uint32_t visible=stripe*bm+sub*rows+row+1;
                const uint32_t offset=sub*words_per_aiv+(stripe*rows+row)*words_per_row;
                for (uint32_t word=0;word<words_per_row;++word) {
                    const uint32_t column=word*32;
                    const uint32_t count=visible>column ? visible-column : 0;
                    masks[offset+word]=count>=32 ? ~uint32_t{0} : (uint32_t{1}<<count)-1;
                }
            }
        }
    }
}

template<uint32_t BM,uint32_t BN> struct Geometry {
    static_assert((BM==32 || BM==64 || BM==128) && (BN==64 || BN==128) && BM<=BN);
    static constexpr uint32_t Rows=BM/2;
    static constexpr uint32_t SBytes=BM*BN*4;
    static constexpr uint32_t PBytes=BM*BN*2;
    static constexpr uint32_t DeltaBytes=BM*D*4;
    static constexpr uint32_t SecondScoreOffset=SBytes+PBytes+DeltaBytes;
    static constexpr uint32_t RelayBytes=RelayBytesFor(BM,BN);
    static constexpr uint32_t CausalMaskWords=Rows*BN/32;
    static constexpr uint32_t CausalMaskBytes=MaskWordsPerAiv(BN)*sizeof(uint32_t);
    static constexpr uint32_t BroadcastBytes=Rows*32;
    static constexpr uint32_t UbBytes=UbScoreSlots*Rows*BN*4+Rows*BN*2+2*Rows*D*4+Rows*D*2+7*Rows*4
        +CausalMaskBytes+BroadcastBytes;
};
constexpr uint16_t QK_READY=0, P_READY=1, PV_READY=2, TILE_DONE=3, QK_EXTRA_READY_BASE=4;
constexpr uint16_t LAST_QK_READY=QK_EXTRA_READY_BASE+ScoreSlots-2;
static_assert(LAST_QK_READY<=15);
} // namespace CubeFA
