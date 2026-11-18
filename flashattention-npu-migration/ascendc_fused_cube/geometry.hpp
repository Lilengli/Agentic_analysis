#pragma once
#include <cstdint>

namespace CubeFA {
constexpr uint32_t D=128;
template<uint32_t BM,uint32_t BN> struct Geometry {
    static_assert((BM==32 || BM==64 || BM==128) && (BN==64 || BN==128) && BM<=BN);
    static constexpr uint32_t Rows=BM/2;
    static constexpr uint32_t SBytes=BM*BN*4;
    static constexpr uint32_t PBytes=BM*BN*2;
    static constexpr uint32_t DeltaBytes=BM*D*4;
    static constexpr uint32_t RelayBytes=SBytes+PBytes+DeltaBytes;
    // Two score-size FP32 buffers: scores and final output/delta are separate.
    static constexpr uint32_t UbBytes=Rows*BN*4+Rows*BN*2+2*Rows*D*4+Rows*D*2+7*Rows*4;
};
constexpr uint16_t QK_READY=0, P_READY=1, PV_READY=2, TILE_DONE=3;
} // namespace CubeFA
