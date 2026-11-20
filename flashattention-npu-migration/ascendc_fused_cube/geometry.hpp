#pragma once
#include <cstdint>

namespace CubeFA {
constexpr uint32_t D=128;
constexpr uint32_t QkLookaheadTiles=2;
constexpr uint32_t ScoreSlots=QkLookaheadTiles+1;
constexpr uint32_t UbScoreSlots=2;
constexpr uint32_t SelectScratchBytes=8*1024;
constexpr uint32_t RelayBytesFor(uint32_t bm, uint32_t bn)
{
    return ScoreSlots * bm * bn * sizeof(float) + bm * bn * sizeof(uint16_t) + bm * D * sizeof(float);
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
    static constexpr uint32_t CausalMaskBytes=(BN/BM)*CausalMaskWords*sizeof(uint32_t);
    static constexpr uint32_t BroadcastBytes=Rows*32;
    static constexpr uint32_t UbBytes=UbScoreSlots*Rows*BN*4+Rows*BN*2+2*Rows*D*4+Rows*D*2+7*Rows*4
        +CausalMaskBytes+BroadcastBytes;
};
constexpr uint16_t QK_READY=0, P_READY=1, PV_READY=2, TILE_DONE=3, QK_NEXT_READY=4, QK_THIRD_READY=5;
} // namespace CubeFA
