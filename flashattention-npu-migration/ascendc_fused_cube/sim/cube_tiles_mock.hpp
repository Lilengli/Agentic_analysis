// CPU semantic model for mixed_kernel.hpp. This checks the mixed-kernel math,
// work partitioning and relay bounds; it does not model Ascend scheduling or
// validate device compilation/lowering.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define __aicore__
#define __global__
#define __mix__(...)
#define __gm__

struct half {
    uint16_t bits = 0;
    half() = default;
    half(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        const uint16_t sign = uint16_t((u >> 16) & 0x8000);
        int e = int((u >> 23) & 255) - 127 + 15;
        uint32_t m = u & 0x7fffff;
        if (((u >> 23) & 255) == 255) {
            bits = uint16_t(sign | 0x7c00 | (m ? 0x0200 : 0));
            return;
        }
        if (e >= 31) {
            bits = uint16_t(sign | 0x7c00);
            return;
        }
        if (e <= 0) {
            if (e < -10) {
                bits = sign;
                return;
            }
            m |= 0x800000;
            const unsigned shift = unsigned(14 - e);
            const uint32_t value = m >> shift;
            const uint32_t rem = m & ((1u << shift) - 1);
            const uint32_t midpoint = 1u << (shift - 1);
            bits = uint16_t(sign | (value + (rem > midpoint || (rem == midpoint && (value & 1)))));
            return;
        }
        m += 0xfff + ((m >> 13) & 1);
        bits = uint16_t(sign | uint16_t(e << 10) + uint16_t(m >> 13));
    }
    operator float() const {
        const unsigned e = (bits >> 10) & 31;
        const unsigned m = bits & 1023;
        const float value = e == 31 ? (m ? NAN : INFINITY)
            : e == 0 ? std::ldexp(float(m), -24)
            : std::ldexp(float(1024 + m), int(e) - 25);
        return (bits & 0x8000) ? -value : value;
    }
    bool operator==(half other) const { return bits == other.bits; }
    bool operator!=(half other) const { return !(*this == other); }
};
static_assert(sizeof(half) == 2);

using GM_ADDR = uint8_t*;
constexpr int PIPE_V = 0;
constexpr int PIPE_S = 1;
constexpr int PIPE_MTE3 = 2;
constexpr int PIPE_FIX = 3;

namespace CubeFASim {
constexpr unsigned kFlagCount = 5;
constexpr unsigned kAivCount = 2;
constexpr std::chrono::seconds kWaitTimeout{5};

struct Stats {
    uint64_t gm_read_elements = 0;
    uint64_t gm_write_elements = 0;
    uint64_t mmad_calls = 0;
    uint64_t whole_reduce_max_calls = 0;
    uint64_t whole_reduce_sum_calls = 0;
    uint64_t max_calls = 0;
    uint64_t adds_calls = 0;
    uint64_t add_calls = 0;
    uint64_t mul_calls = 0;
    uint64_t muls_calls = 0;
    uint64_t exp_calls = 0;
    uint64_t cast_calls = 0;
    uint64_t duplicate_calls = 0;
    std::array<size_t, 5> local_bytes_by_position{};
};

enum class Core { AIC, AIV };

struct Context {
    uint32_t group = 0;
    Core core = Core::AIC;
    uint32_t sub = 0;
    Stats* stats = nullptr;
    std::array<uint64_t, kFlagCount> waited_arrivals{};
    std::array<uint64_t, kFlagCount> seen_broadcasts{};
};
inline thread_local Context context{};

struct FlagState {
    std::mutex mutex;
    std::condition_variable changed;
    std::array<uint64_t, kFlagCount> broadcasts{};
    std::array<uint64_t, kFlagCount> arrivals{};
    bool aborted = false;
};
inline std::mutex groups_mutex;
inline std::vector<std::unique_ptr<FlagState>> groups;

inline uintptr_t workspace_begin = 0;
inline uintptr_t workspace_end = 0;
inline size_t workspace_stride = 0;
inline uint32_t workspace_groups = 0;

inline void SetContext(uint32_t group, Core core, uint32_t sub, Stats* stats) {
    context = Context{};
    context.group = group;
    context.core = core;
    context.sub = sub;
    context.stats = stats;
}

inline void BeginRun(uint32_t group_count, const void* relay, size_t relay_stride) {
    std::lock_guard<std::mutex> lock(groups_mutex);
    groups.clear();
    groups.reserve(group_count);
    for (uint32_t i = 0; i < group_count; ++i) groups.emplace_back(new FlagState);
    workspace_begin = reinterpret_cast<uintptr_t>(relay);
    workspace_stride = relay_stride;
    workspace_groups = group_count;
    workspace_end = workspace_begin + size_t(group_count) * relay_stride;
}

inline FlagState& GroupFlags(uint32_t group) {
    if (group >= groups.size()) throw std::runtime_error("mixed simulator: invalid group index");
    return *groups[group];
}

inline void AbortGroup(uint32_t group) noexcept {
    try {
        FlagState& state = GroupFlags(group);
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.aborted = true;
        }
        state.changed.notify_all();
    } catch (...) {}
}

inline void SetFlag(uint32_t flag) {
    if (flag >= kFlagCount) throw std::runtime_error("mixed simulator: invalid flag");
    FlagState& state = GroupFlags(context.group);
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (context.core == Core::AIC) ++state.broadcasts[flag];
        else ++state.arrivals[flag];
    }
    state.changed.notify_all();
}

inline void WaitFlag(uint32_t flag) {
    if (flag >= kFlagCount) throw std::runtime_error("mixed simulator: invalid flag");
    FlagState& state = GroupFlags(context.group);
    std::unique_lock<std::mutex> lock(state.mutex);
    if (context.core == Core::AIV) {
        const uint64_t target = context.seen_broadcasts[flag] + 1;
        if (!state.changed.wait_for(lock, kWaitTimeout, [&] {
                return state.aborted || state.broadcasts[flag] >= target;
            })) {
            state.aborted = true;
            state.changed.notify_all();
            throw std::runtime_error("mixed simulator: timed out waiting for AIC broadcast flag " + std::to_string(flag));
        }
        if (state.aborted) throw std::runtime_error("mixed simulator: group aborted while waiting for AIC broadcast");
        ++context.seen_broadcasts[flag];
    } else {
        const uint64_t target = (context.waited_arrivals[flag] + 1) * kAivCount;
        if (!state.changed.wait_for(lock, kWaitTimeout, [&] {
                return state.aborted || state.arrivals[flag] >= target;
            })) {
            state.aborted = true;
            state.changed.notify_all();
            throw std::runtime_error("mixed simulator: timed out waiting for both AIV arrivals on flag " + std::to_string(flag));
        }
        if (state.aborted) throw std::runtime_error("mixed simulator: group aborted while waiting for AIV arrivals");
        ++context.waited_arrivals[flag];
    }
}

inline void CheckWorkspaceWrite(const void* destination, size_t bytes) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(destination);
    const uintptr_t end = begin + bytes;
    if (begin >= workspace_begin && begin < workspace_end) {
        const size_t group = (begin - workspace_begin) / workspace_stride;
        const uintptr_t group_begin = workspace_begin + group * workspace_stride;
        const uintptr_t group_end = group_begin + workspace_stride;
        if (group >= workspace_groups || group != context.group || end > group_end)
            throw std::runtime_error("mixed simulator: relay write escaped its group's workspace");
    }
}

inline void CountRead(size_t elements) {
    if (context.stats) context.stats->gm_read_elements += elements;
}
inline void CountWrite(const void* destination, size_t elements, size_t element_size) {
    CheckWorkspaceWrite(destination, elements * element_size);
    if (context.stats) context.stats->gm_write_elements += elements;
}
inline void CountMmad() { if (context.stats) ++context.stats->mmad_calls; }
inline void CountVector(uint64_t Stats::*member) { if (context.stats) ++(context.stats->*member); }
inline void CountBuffer(unsigned position, size_t bytes) {
    if (position >= 5) throw std::runtime_error("mixed simulator: invalid TBuf position");
    if (context.stats) context.stats->local_bytes_by_position[position] += bytes;
}
} // namespace CubeFASim

namespace AscendC {
enum { AIC = 0, AIV = 1 };
enum class TPosition { A1 = 0, A2 = 1, B2 = 2, CO1 = 3, VECCALC = 4 };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class ReduceOrder { ORDER_ONLY_VALUE };
enum class HardEvent { V_MTE2, MTE2_V, V_S, S_V, V_MTE3, MTE3_V, FIX_M };

inline uint32_t GetBlockIdx() { return CubeFASim::context.group * 3; }
inline uint32_t GetBlockNum() { return uint32_t(CubeFASim::groups.size()) * 3; }
inline uint32_t GetSubBlockNum() { return 3; }
inline uint32_t GetSubBlockIdx() { return CubeFASim::context.sub; }

inline thread_local bool l0c_slot_ready[2]{};

template <HardEvent E> void SetFlag(int id) {
    if constexpr (E == HardEvent::FIX_M) {
        if (id < 0 || id >= 2 || l0c_slot_ready[id])
            throw std::runtime_error("mixed simulator: invalid L0C slot release");
        l0c_slot_ready[id] = true;
    }
}
template <HardEvent E> void WaitFlag(int id) {
    if constexpr (E == HardEvent::FIX_M) {
        if (id < 0 || id >= 2 || !l0c_slot_ready[id])
            throw std::runtime_error("mixed simulator: L0C slot has no release");
        l0c_slot_ready[id] = false;
    }
}
template <int> void PipeBarrier() {}
template <int Mode, int Pipe> void CrossCoreSetFlag(uint32_t flag) { CubeFASim::SetFlag(flag); }
inline void CrossCoreWaitFlag(uint32_t flag) { CubeFASim::WaitFlag(flag); }

template <class T> struct LocalTensor {
    std::shared_ptr<std::vector<uint8_t>> owner;
    size_t offset = 0;
    size_t length = 0;
    T GetValue(size_t i) const {
        if (!owner || i >= length) throw std::runtime_error("mixed simulator: UB read out of bounds");
        T value;
        std::memcpy(&value, owner->data() + offset + i * sizeof(T), sizeof(T));
        return value;
    }
    void SetValue(size_t i, T value) const {
        if (!owner || i >= length) throw std::runtime_error("mixed simulator: UB write out of bounds");
        std::memcpy(owner->data() + offset + i * sizeof(T), &value, sizeof(T));
    }
    LocalTensor operator[](size_t i) const {
        if (i > length) throw std::runtime_error("mixed simulator: UB slice out of bounds");
        return {owner, offset + i * sizeof(T), length - i};
    }
};

template <class T> struct GlobalTensor {
    T* data = nullptr;
    size_t length = 0;
    size_t offset = 0;
    void SetGlobalBuffer(T* pointer, size_t elements) { data = pointer; length = elements; offset = 0; }
    GlobalTensor operator[](size_t i) const {
        if (i > length) throw std::runtime_error("mixed simulator: GM slice out of bounds");
        return {data + i, length - i, offset + i * sizeof(T)};
    }
};

template <TPosition P> struct TBuf {
    std::shared_ptr<std::vector<uint8_t>> storage;
    template <class T> LocalTensor<T> Get() {
        if (!storage || storage->size() % sizeof(T))
            throw std::runtime_error("mixed simulator: invalid TBuf reinterpretation");
        return {storage, 0, storage->size() / sizeof(T)};
    }
};

struct TPipe {
    template <TPosition P> void InitBuffer(TBuf<P>& buffer, size_t bytes) {
        if (bytes % 32) throw std::runtime_error("mixed simulator: TBuf size is not 32-byte aligned");
        buffer.storage = std::make_shared<std::vector<uint8_t>>(bytes, 0xff);
        CubeFASim::CountBuffer(unsigned(P), bytes);
    }
    int FetchEventID(HardEvent) { return 0; }
};

template <class T> void DataCopy(LocalTensor<T> destination, GlobalTensor<T> source, size_t n) {
    if (destination.offset % 32 || source.offset % 32 || (n * sizeof(T)) % 32)
        throw std::runtime_error("mixed simulator: unaligned GM-to-UB DataCopy");
    if (n > destination.length || n > source.length)
        throw std::runtime_error("mixed simulator: GM-to-UB DataCopy out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, source.data[i]);
    CubeFASim::CountRead(n);
}
template <class T> void DataCopy(GlobalTensor<T> destination, LocalTensor<T> source, size_t n) {
    if (destination.offset % 32 || source.offset % 32 || (n * sizeof(T)) % 32)
        throw std::runtime_error("mixed simulator: unaligned UB-to-GM DataCopy");
    if (n > destination.length || n > source.length)
        throw std::runtime_error("mixed simulator: UB-to-GM DataCopy out of bounds");
    CubeFASim::CountWrite(destination.data, n, sizeof(T));
    for (size_t i = 0; i < n; ++i) destination.data[i] = source.GetValue(i);
}

template <class A, class B> void Cast(LocalTensor<A> destination, LocalTensor<B> source, RoundMode, size_t n) {
    if (n > destination.length || n > source.length) throw std::runtime_error("mixed simulator: Cast out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, static_cast<A>(source.GetValue(i)));
    CubeFASim::CountVector(&CubeFASim::Stats::cast_calls);
}
template <class T> void Duplicate(LocalTensor<T> destination, T value, size_t n) {
    if (n > destination.length) throw std::runtime_error("mixed simulator: Duplicate out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, value);
    CubeFASim::CountVector(&CubeFASim::Stats::duplicate_calls);
}
inline void Duplicate(LocalTensor<float> destination, float value, const uint64_t mask[],
        uint8_t repeats, uint16_t block_stride, uint8_t repeat_stride) {
    constexpr size_t block_elements = 32 / sizeof(float);
    if (destination.offset % 32 || !mask[0] || mask[1] || !repeats)
        throw std::runtime_error("mixed simulator: invalid masked FP32 Duplicate");
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        for (size_t lane = 0; lane < 64; ++lane) {
            if (!(mask[0] & (uint64_t{1} << lane))) continue;
            const size_t block = repeat * repeat_stride + lane / block_elements * block_stride;
            destination.SetValue(block * block_elements + lane % block_elements, value);
        }
    }
    CubeFASim::CountVector(&CubeFASim::Stats::duplicate_calls);
}

template <class T> void WholeReduceMax(LocalTensor<T> destination, LocalTensor<T> source,
        int mask, int repeats, int dstStride, int blockStride, int repStride, ReduceOrder) {
    constexpr int block = 32 / sizeof(T);
    if (mask <= 0 || repeats < 0) throw std::runtime_error("mixed simulator: invalid WholeReduceMax shape");
    for (int r = 0; r < repeats; ++r) {
        T value = source.GetValue(size_t(r * repStride * block));
        for (int i = 1; i < mask; ++i) {
            const size_t index = size_t(r * repStride * block + (i / block) * blockStride * block + i % block);
            value = std::max(value, source.GetValue(index));
        }
        destination.SetValue(size_t(r * dstStride), value);
    }
    CubeFASim::CountVector(&CubeFASim::Stats::whole_reduce_max_calls);
}
template <class T> void WholeReduceSum(LocalTensor<T> destination, LocalTensor<T> source,
        int mask, int repeats, int dstStride, int blockStride, int repStride) {
    constexpr int block = 32 / sizeof(T);
    if (mask <= 0 || repeats < 0) throw std::runtime_error("mixed simulator: invalid WholeReduceSum shape");
    for (int r = 0; r < repeats; ++r) {
        T value = T(0);
        for (int i = 0; i < mask; ++i) {
            const size_t index = size_t(r * repStride * block + (i / block) * blockStride * block + i % block);
            value += source.GetValue(index);
        }
        destination.SetValue(size_t(r * dstStride), value);
    }
    CubeFASim::CountVector(&CubeFASim::Stats::whole_reduce_sum_calls);
}
template <class T> void Max(LocalTensor<T> destination, LocalTensor<T> a, LocalTensor<T> b, size_t n) {
    if (n > destination.length || n > a.length || n > b.length) throw std::runtime_error("mixed simulator: Max out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, std::max(a.GetValue(i), b.GetValue(i)));
    CubeFASim::CountVector(&CubeFASim::Stats::max_calls);
}
template <class T> void Adds(LocalTensor<T> destination, LocalTensor<T> source, T scalar, size_t n) {
    if (n > destination.length || n > source.length) throw std::runtime_error("mixed simulator: Adds out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, source.GetValue(i) + scalar);
    CubeFASim::CountVector(&CubeFASim::Stats::adds_calls);
}
template <class T> void Add(LocalTensor<T> destination, LocalTensor<T> a, LocalTensor<T> b, size_t n) {
    if (n > destination.length || n > a.length || n > b.length) throw std::runtime_error("mixed simulator: Add out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, a.GetValue(i) + b.GetValue(i));
    CubeFASim::CountVector(&CubeFASim::Stats::add_calls);
}
template <class T> void Mul(LocalTensor<T> destination, LocalTensor<T> a, LocalTensor<T> b, size_t n) {
    if (n > destination.length || n > a.length || n > b.length) throw std::runtime_error("mixed simulator: Mul out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, a.GetValue(i) * b.GetValue(i));
    CubeFASim::CountVector(&CubeFASim::Stats::mul_calls);
}
template <class T> void Muls(LocalTensor<T> destination, LocalTensor<T> source, T scalar, size_t n) {
    if (n > destination.length || n > source.length) throw std::runtime_error("mixed simulator: Muls out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, source.GetValue(i) * scalar);
    CubeFASim::CountVector(&CubeFASim::Stats::muls_calls);
}
template <class T> void Exp(LocalTensor<T> destination, LocalTensor<T> source, size_t n) {
    if (n > destination.length || n > source.length) throw std::runtime_error("mixed simulator: Exp out of bounds");
    for (size_t i = 0; i < n; ++i) destination.SetValue(i, std::exp(source.GetValue(i)));
    CubeFASim::CountVector(&CubeFASim::Stats::exp_calls);
}
} // namespace AscendC

namespace Catlass {
namespace layout {
struct RowMajor {
    size_t rows, cols, stride;
    RowMajor(size_t row_count, size_t col_count) : rows(row_count), cols(col_count), stride(col_count) {}
    RowMajor(size_t row_count, size_t col_count, size_t row_stride) : rows(row_count), cols(col_count), stride(row_stride) {}
    size_t Offset(size_t row, size_t col) const { return row * stride + col; }
};
struct ColumnMajor {
    size_t rows, cols, stride;
    ColumnMajor(size_t row_count, size_t col_count, size_t col_stride)
        : rows(row_count), cols(col_count), stride(col_stride) {}
    size_t Offset(size_t row, size_t col) const { return row + col * stride; }
};
} // namespace layout
} // namespace Catlass

namespace CubeFA {
template <AscendC::HardEvent E>
__aicore__ inline void Sync(AscendC::TPipe& pipe) {
    const int id = pipe.FetchEventID(E);
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

template<bool DoubleBufferedL0C=false, class ALayout, class BLayout>
__aicore__ inline void Gemm(AscendC::TPipe&, AscendC::GlobalTensor<half> a,
        AscendC::GlobalTensor<half> b, AscendC::GlobalTensor<float> c,
        AscendC::LocalTensor<half>, AscendC::LocalTensor<half>,
        AscendC::LocalTensor<half>, AscendC::LocalTensor<half>,
        AscendC::LocalTensor<float> l0c, uint32_t m, uint32_t n, uint32_t k,
        ALayout a_layout, BLayout b_layout, bool = true, int32_t l0c_slot = 0) {
    if constexpr (DoubleBufferedL0C) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
        const size_t slot_bytes = l0c.owner->size() / 2;
        if (l0c.offset != size_t(l0c_slot) * slot_bytes || size_t(m) * n * sizeof(float) > slot_bytes)
            throw std::runtime_error("mixed simulator: L0C tensor does not match its slot");
    }
    if (a_layout.rows != m || a_layout.cols != k || b_layout.rows != k || b_layout.cols != n)
        throw std::runtime_error("mixed simulator: GEMM layout does not match its dimensions");
    if (a.offset % 32 || b.offset % 32 || c.offset % 32 ||
        (size_t(m) * n * sizeof(float)) % 32)
        throw std::runtime_error("mixed simulator: GEMM operand/output alignment is invalid");
    const size_t last_a = a_layout.Offset(m - 1, k - 1) + 1;
    const size_t last_b = b_layout.Offset(k - 1, n - 1) + 1;
    if (a.length < last_a || b.length < last_b || c.length < size_t(m) * n)
        throw std::runtime_error("mixed simulator: GEMM operand or output out of bounds");
    CubeFASim::CountRead(size_t(m) * k + size_t(k) * n);
    CubeFASim::CountWrite(c.data, size_t(m) * n, sizeof(float));
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (uint32_t x = 0; x < k; ++x)
                sum += float(a.data[a_layout.Offset(i, x)]) * float(b.data[b_layout.Offset(x, j)]);
            c.data[size_t(i) * n + j] = sum;
        }
    }
    CubeFASim::CountMmad();
    if constexpr (DoubleBufferedL0C) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0c_slot);
    }
}
} // namespace CubeFA
