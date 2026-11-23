#pragma once
#include <cstdint>

namespace CubeFA {
constexpr uint32_t D=128;
constexpr uint32_t PvRowTile=32, PvFeatureTile=32, PvFullKv=512, PvBSlots=2;

// Four compact 32x32 NZ matrices per query stripe. In GM these are
// scattered to successive 32-column bands of a row-major 32x128 stripe.
struct PvBatchFixLayout {
    static constexpr uint32_t Tiles=D/PvFeatureTile;
    static constexpr uint32_t TileElements=PvRowTile*PvFeatureTile;
    static constexpr uint32_t StripeElements=PvRowTile*D;
    static constexpr uint32_t NzColumns=16;
    static constexpr uint32_t NzFractalBytes=NzColumns*NzColumns*sizeof(float);
    static constexpr uint32_t SrcStride=PvRowTile; // in C0-sized rows
    static constexpr uint32_t SrcMatrixStride=TileElements*sizeof(float)/NzFractalBytes;
    static constexpr uint32_t DstStride=D; // in elements
    static constexpr uint32_t DstMatrixStride=PvFeatureTile; // in elements
};
static_assert(PvBatchFixLayout::Tiles==4 && sizeof(float)==4);
// Bound the QK work that can sit ahead of a ready PV on the same pipelines.
constexpr uint32_t ScoreSlots=4;
constexpr uint32_t QkLookaheadTiles=ScoreSlots-1;
constexpr uint32_t PvRelaySlots=3;
constexpr uint32_t UbScoreSlots=2;
static_assert(PvRelaySlots>=2 && QkLookaheadTiles>=PvRelaySlots-1);
constexpr uint32_t SelectScratchBytes=8*1024;
constexpr uint32_t RelayBytesFor(uint32_t bm, uint32_t bn)
{
    return ScoreSlots * bm * bn * sizeof(float) +
        PvRelaySlots * (bm * bn * sizeof(uint16_t) + bm * D * sizeof(float));
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
    static_assert(((BM==32 || BM==64 || BM==128) && (BN==64 || BN==128) && BM<=BN)
        || (BM==128 && BN==512));
    static constexpr uint32_t Rows=BM/2;
    static constexpr uint32_t CubeKv=BN==512 ? 128 : BN;
    static constexpr uint32_t BatchRows=BN==512 ? 16 : Rows;
    static constexpr uint32_t BatchElements=BatchRows*BN;
    static constexpr uint32_t PBufferSlots=BN==512 ? 1 : PvRelaySlots-1;
    static constexpr uint32_t SBytes=BM*BN*4;
    static constexpr uint32_t PBytes=BM*BN*2;
    static constexpr uint32_t DeltaBytes=BM*D*4;
    static constexpr uint32_t SecondScoreOffset=SBytes+PvRelaySlots*(PBytes+DeltaBytes);
    static constexpr uint32_t RelayBytes=RelayBytesFor(BM,BN);
    static constexpr uint32_t CausalMaskWords=Rows*BN/32;
    static constexpr uint32_t CausalMaskBytes=MaskWordsPerAiv(BN)*sizeof(uint32_t);
    static constexpr uint32_t BroadcastBytes=Rows*32;
    static constexpr uint32_t UbBytes=UbScoreSlots*BatchElements*4+PBufferSlots*BatchElements*2
        +2*Rows*D*4+Rows*D*2+(6+PvRelaySlots)*Rows*4+CausalMaskBytes+BroadcastBytes;
};
// Ordered QK notifications alternate two counters independently of the GM ring.
constexpr uint16_t QK_READY=0, P_READY=1, PV_READY=2, TILE_DONE=3, QK_ODD_READY=4;
constexpr uint16_t RelayExtraBase=5, FlagsPerRelaySlot=TILE_DONE-P_READY+1;
constexpr uint16_t LAST_RELAY_FLAG=RelayExtraBase+(PvRelaySlots-1)*FlagsPerRelaySlot-1;
static_assert(ScoreSlots%2==0 && LAST_RELAY_FLAG<=10);
} // namespace CubeFA
