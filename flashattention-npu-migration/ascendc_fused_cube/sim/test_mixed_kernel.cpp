#include "../mixed_kernel.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
template<unsigned BM, unsigned BN, bool PvRow32>
constexpr uint64_t MmadPerStack = BN / CubeFA::Geometry<BM, BN>::CubeKv +
    (PvRow32 ? (BM / CubeFA::PvRowTile) * (CubeFA::D / CubeFA::PvFeatureTile)
             : BN / CubeFA::Geometry<BM, BN>::CubeKv);

constexpr uint8_t kRelayGuard = 0x6d;
constexpr size_t kGuardBytes = 64;

template<int Core, unsigned BM, unsigned BN, bool PvRow32=false, bool BatchFix=false, bool PrioritizeP=false, bool DeferQkFixpipe=false>
void RunCore(uint32_t group, uint32_t sub, CubeFASim::Stats& stats,
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR relay, GM_ADDR masks,
        uint32_t heads, uint32_t seq, uint32_t groups) {
    CubeFASim::SetContext(group, Core == AscendC::AIC ? CubeFASim::Core::AIC : CubeFASim::Core::AIV, sub, &stats);
    CubeFA::RunCore<Core, BM, BN, PvRow32, BatchFix, PrioritizeP, DeferQkFixpipe>(q, k, v, out, relay, masks, heads, seq, groups);
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint64_t AssignedTasks(uint32_t group, uint32_t groups, uint64_t task_count) {
    uint64_t count = 0;
    for (uint64_t task = group; task < task_count; task += groups) ++count;
    return count;
}

template<unsigned BM, unsigned BN>
uint64_t ExpectedIterations(uint32_t group, uint32_t groups, uint32_t heads, uint32_t seq) {
    const uint32_t tiles = seq / BM;
    const uint64_t task_count = uint64_t(heads) * tiles;
    uint64_t count = 0;
    for (uint64_t task = group; task < task_count; task += groups) {
        const uint32_t qs = uint32_t(task % tiles) * BM;
        count += qs / BN + 1;
    }
    return count;
}

template<unsigned BM, unsigned BN, bool PvRow32=false, bool BatchFix=false, bool PrioritizeP=false, bool DeferQkFixpipe=false>
void CheckStats(uint32_t group, uint32_t groups, uint32_t heads, uint32_t seq,
        const CubeFASim::Stats& aic, const CubeFASim::Stats& av0,
        const CubeFASim::Stats& av1) {
    using G = CubeFA::Geometry<BM, BN>;
    constexpr uint32_t R = G::Rows;
    constexpr uint32_t MaxK = G::CubeKv > CubeFA::D ? G::CubeKv : CubeFA::D;
    constexpr uint32_t MaxN = MaxK;
    constexpr uint64_t p_pulses = PvRow32 ? 2 : 1;
    constexpr uint64_t batches = R / G::BatchRows;
    constexpr uint64_t reductions_per_tile = batches * (BN == 512 ? 3 : (BN / 64) * 2);
    const uint64_t tasks = AssignedTasks(group, groups, uint64_t(heads) * (seq / BM));
    const uint64_t iterations = ExpectedIterations<BM, BN>(group, groups, heads, seq);
    std::array<uint64_t, CubeFA::ScoreSlots + 1> expected_pending{};
    std::array<uint64_t, CubeFA::ScoreSlots + 1> expected_pending_before_done{};
    std::vector<uint64_t> expected_prepared_at_pv_wait;
    const uint32_t query_tiles = seq / BM;
    uint64_t prepared = 0;
    // The experiment primes exactly two score stacks; each refill follows PV.
    constexpr uint32_t qk_pending_cap = (PrioritizeP || DeferQkFixpipe) ? 2 : CubeFA::ScoreSlots;
    for (uint64_t task = group; task < uint64_t(heads) * query_tiles; task += groups) {
        const uint32_t kv_tiles = uint32_t(task % query_tiles) * BM / BN + 1;
        for (uint32_t remaining = kv_tiles; remaining > 0; --remaining) {
            ++expected_pending[std::min(remaining, qk_pending_cap)];
            if constexpr (PvRow32) ++expected_pending[std::min(remaining, qk_pending_cap)];
        }
        for (uint32_t tile = CubeFA::PvRelaySlots; tile < kv_tiles; ++tile)
            ++expected_pending_before_done[
                std::min(kv_tiles - tile, qk_pending_cap)];
        expected_pending_before_done[0] += std::min(kv_tiles, CubeFA::PvRelaySlots);
        for (uint32_t tile = 0; tile < kv_tiles; ++tile)
            expected_prepared_at_pv_wait.push_back(
                prepared + std::min(tile + CubeFA::PvRelaySlots, kv_tiles));
        prepared += kv_tiles;
    }
    Require(aic.pending_qk_at_p_waits == expected_pending,
        "AIC did not fill the expected score slots before waiting for P");
    Require(aic.pending_qk_at_done_waits == expected_pending_before_done,
        "AIC TILE_DONE waits do not match relay slot reuse and final drain");
    Require(aic.mmad_calls == iterations * MmadPerStack<BM, BN, PvRow32>,
        "AIC group " + std::to_string(group) + " did not issue all QK and PV slices");
    if constexpr (PrioritizeP || DeferQkFixpipe) {
        Require(aic.p_ready_mmad_count.size() == iterations,
            "AIC must wait exactly once for each full P stack");
        size_t sample = 0;
        uint64_t total_mmad = 0, total_fix = 0;
        constexpr uint32_t slices = BN/G::CubeKv;
        for (uint64_t task = group; task < uint64_t(heads) * query_tiles; task += groups) {
            const uint32_t stacks = uint32_t(task % query_tiles) * BM / BN + 1;
            const uint32_t primed = std::min(stacks, 2u)*slices;
            total_mmad += primed;
            total_fix += primed;
            for (uint32_t stack = 0; stack < stacks; ++stack) {
                const bool refill = stack+2 < stacks;
                const uint32_t staged = DeferQkFixpipe && refill ? 2 : 0;
                Require(aic.p_ready_mmad_count[sample] == total_mmad+staged,
                    "AIC did not stage the expected number of QK slices before waiting for P");
                Require(aic.p_ready_fixpipe_count[sample++] == total_fix,
                    "AIC wrote a newer score slice before waiting for P");
                total_mmad += slices + (refill ? slices : 0);
                total_fix += 1 + (refill ? slices : 0);
            }
        }
    }
    constexpr uint64_t pv_fixpipe_calls = !PvRow32 ? 1
        : (BM/CubeFA::PvRowTile) * (BatchFix ? 1 : CubeFA::D/CubeFA::PvFeatureTile);
    Require(aic.fixpipe_calls == iterations * (BN/G::CubeKv + pv_fixpipe_calls),
        "AIC did not issue the expected QK and PV Fixpipe calls");
    if constexpr (PvRow32) {
        const uint64_t qk_reads = (BN / G::CubeKv) * (BM * CubeFA::D + G::CubeKv * CubeFA::D);
        const uint64_t pv_reads = BM * BN + BN * CubeFA::D;
        Require(aic.gm_read_elements == iterations * (qk_reads + pv_reads),
            "AIC must load each P and V element once per PV stack");
        Require(aic.p_ready_mmad_count.size() == 2 * iterations,
            "AIC must receive two probability tranches per stack");
        for (size_t i = 0; i < aic.p_ready_mmad_count.size(); i += 2)
            Require(aic.p_ready_mmad_count[i+1] - aic.p_ready_mmad_count[i] ==
                    2 * CubeFA::D / CubeFA::PvFeatureTile,
                "first PV tranche must be computed before waiting for the second");
    }
    Require(aic.gm_write_elements == iterations * (uint64_t(BM) * BN + uint64_t(BM) * CubeFA::D),
        "AIC score/delta workspace write count is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::A1)] ==
                2 * (uint64_t(BM) * CubeFA::D +
                (PvRow32 ? 2 * CubeFA::PvRowTile * BN : BM * G::CubeKv) +
                (PvRow32 ? BN : G::CubeKv) * CubeFA::D),
        "AIC A1 buffer sizes do not match the mixed schedule");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::A2)] == 2 * (PvRow32 ? 2 * CubeFA::PvRowTile * BN : BM * MaxK),
        "AIC A2 buffer size is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::B2)] ==
                2 * (PvRow32 ? CubeFA::PvBSlots * BN * CubeFA::PvFeatureTile : G::CubeKv * CubeFA::D),
        "AIC B2 buffer size is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::CO1)] == 2 * uint64_t(BM) * MaxN * 4,
        "AIC CO1 buffer size is incorrect");

    for (const CubeFASim::Stats* av : {&av0, &av1}) {
        Require(av->p_ready_calls == iterations * p_pulses,
            "AIV probability notification count is incorrect");
        constexpr uint64_t mul_calls_per_tile = batches + CubeFA::D / 64;
        const uint64_t mask_reads = tasks ? CubeFA::MaskWordsPerAiv(BN) : 0;
        Require(av->reads_before_first_qk == mask_reads,
            "AIV must load its packed mask before the first QK wait");
        Require(av->local_bytes_by_position[unsigned(AscendC::TPosition::VECCALC)] == G::UbBytes,
            "AIV UB buffers exceed or differ from Geometry::UbBytes");
        Require(av->pv_waits.size() == expected_prepared_at_pv_wait.size(),
            "AIV PV wait count does not match the number of KV tiles");
        for (size_t i = 0; i < av->pv_waits.size(); ++i) {
            const uint64_t scores_prepared = expected_prepared_at_pv_wait[i];
            const auto& snapshot = av->pv_waits[i];
            Require(snapshot.p_ready_calls == scores_prepared * p_pulses,
                "AIV must publish its full probability lookahead before waiting for the current PV");
            Require(snapshot.gm_read_elements == mask_reads + (scores_prepared * BN + i * CubeFA::D) * R,
                "AIV did not prefetch the next score before waiting for PV");
            Require(snapshot.block_reduce_max_calls == scores_prepared * reductions_per_tile,
                "AIV did not reduce the next score before waiting for PV");
            Require(snapshot.exp_calls == 2 * batches * scores_prepared &&
                    snapshot.block_reduce_sum_calls == scores_prepared * reductions_per_tile,
                "AIV must finish the next softmax before waiting for PV");
            Require(snapshot.mul_calls == batches * scores_prepared + (i + 1) * (CubeFA::D / 64),
                "AIV must rescale the denominator and output before waiting for PV");
        }
        Require(av->gm_read_elements == mask_reads + iterations * (uint64_t(R) * BN + uint64_t(R) * CubeFA::D),
            "AIV mask/score/delta GM read count is incorrect");
        Require(av->gm_write_elements == iterations * uint64_t(R) * BN + tasks * uint64_t(R) * CubeFA::D,
            "AIV probability/output GM write count is incorrect");
        const uint64_t reduce_calls = iterations * reductions_per_tile;
        Require(av->block_reduce_max_calls == reduce_calls && av->block_reduce_sum_calls == reduce_calls,
            "AIV reduction calls do not match the score tile width");
        Require(av->max_calls == iterations * batches * (BN == 128 ? 2 : 1),
            "AIV must update the running maxima as vectors");
        Require(av->brcb_calls == (batches + 1) * iterations &&
                    av->sub_calls == iterations * batches * (1 + BN / 64),
            "AIV must broadcast both the updated maxima and the output rescale factors");
        Require(av->mul_calls == mul_calls_per_tile * iterations && av->exp_calls == 2 * batches * iterations,
            "AIV online-softmax vector operation counts are incorrect");
        Require(av->cast_calls == batches * iterations + tasks,
            "AIV probability or output conversion count is incorrect");
        Require(av->duplicate_calls == 3 * tasks && av->select_calls == batches * tasks,
            "AIV must initialize online state and select the packed mask per diagonal batch");
    }
}

template<unsigned BM, unsigned BN, bool PvRow32=false, bool BatchFix=false, bool PrioritizeP=false, bool DeferQkFixpipe=false>
void Check(unsigned seq, unsigned heads, unsigned groups, unsigned repeat) {
    std::mt19937 gen(0xC0BFAu + seq * 17 + heads * 31 + BM * 7 + BN + repeat * 131);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    const size_t elements = size_t(heads) * seq * CubeFA::D;
    std::vector<half> q(elements), k(elements), v(elements), out(elements, half(NAN));
    for (std::vector<half>* tensor : {&q, &k, &v})
        for (half& value : *tensor) value = half(distribution(gen));
    const std::vector<half> original_q = q, original_k = k, original_v = v;

    using G = CubeFA::Geometry<BM, BN>;
    std::vector<uint32_t> masks(2 * CubeFA::MaskWordsPerAiv(BN));
    CubeFA::PackCausalMasks(masks.data(), BM, BN);
    GM_ADDR mask_base = reinterpret_cast<GM_ADDR>(masks.data());
    std::vector<uint8_t> relay(kGuardBytes + size_t(groups) * G::RelayBytes + kGuardBytes, kRelayGuard);
    GM_ADDR relay_base = relay.data() + kGuardBytes;
    std::fill(relay_base, relay_base + size_t(groups) * G::RelayBytes, uint8_t(0xff));
    CubeFASim::p_pulses_per_stack = PvRow32 ? 2 : 1;
    CubeFASim::BeginRun(groups, relay_base, G::RelayBytes);

    std::vector<CubeFASim::Stats> aic_stats(groups), aiv0_stats(groups), aiv1_stats(groups);
    std::vector<std::thread> workers;
    workers.reserve(size_t(groups) * 3);
    std::mutex error_mutex;
    std::string first_error;
    auto record_error = [&](uint32_t group, const std::exception& error) {
        CubeFASim::AbortGroup(group);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error.empty()) first_error = error.what();
    };
    for (uint32_t group = 0; group < groups; ++group) {
        workers.emplace_back([&, group] {
            try {
                RunCore<AscendC::AIC, BM, BN, PvRow32, BatchFix, PrioritizeP, DeferQkFixpipe>(group, 0, aic_stats[group],
                    reinterpret_cast<GM_ADDR>(q.data()), reinterpret_cast<GM_ADDR>(k.data()),
                    reinterpret_cast<GM_ADDR>(v.data()), reinterpret_cast<GM_ADDR>(out.data()),
                    relay_base, mask_base, heads, seq, groups);
            } catch (const std::exception& e) { record_error(group, e); }
        });
        for (uint32_t sub = 0; sub < 2; ++sub) {
            workers.emplace_back([&, group, sub] {
                try {
                    CubeFASim::Stats& stats = sub == 0 ? aiv0_stats[group] : aiv1_stats[group];
                    RunCore<AscendC::AIV, BM, BN, PvRow32, BatchFix, PrioritizeP, DeferQkFixpipe>(group, sub, stats,
                        reinterpret_cast<GM_ADDR>(q.data()), reinterpret_cast<GM_ADDR>(k.data()),
                        reinterpret_cast<GM_ADDR>(v.data()), reinterpret_cast<GM_ADDR>(out.data()),
                        relay_base, mask_base, heads, seq, groups);
                } catch (const std::exception& e) { record_error(group, e); }
            });
        }
    }
    for (std::thread& worker : workers) worker.join();
    if (!first_error.empty()) throw std::runtime_error("mixed kernel worker failed: " + first_error);

    Require(q == original_q && k == original_k && v == original_v, "kernel modified an input tensor");
    for (size_t i = 0; i < kGuardBytes; ++i) {
        Require(relay[i] == kRelayGuard, "relay write crossed the leading guard");
        Require(relay[kGuardBytes + size_t(groups) * G::RelayBytes + i] == kRelayGuard,
            "relay write crossed the trailing guard");
    }

    uint64_t total_iterations = 0;
    uint64_t total_mmad_calls = 0;
    uint64_t total_output_writes = 0;
    for (uint32_t group = 0; group < groups; ++group) {
        CheckStats<BM, BN, PvRow32, BatchFix, PrioritizeP, DeferQkFixpipe>(group, groups, heads, seq, aic_stats[group], aiv0_stats[group], aiv1_stats[group]);
        const uint64_t iterations = ExpectedIterations<BM, BN>(group, groups, heads, seq);
        total_iterations += iterations;
        total_mmad_calls += aic_stats[group].mmad_calls;
        total_output_writes += aiv0_stats[group].gm_write_elements - iterations * (BM / 2) * BN;
        total_output_writes += aiv1_stats[group].gm_write_elements - iterations * (BM / 2) * BN;
    }
    Require(total_mmad_calls == total_iterations * MmadPerStack<BM, BN, PvRow32>,
        "total AIC Mmad count does not match the QK and PV slices");
    Require(total_output_writes == elements, "output work partition left rows unwritten or wrote rows more than once");

    double max_error = 0.0;
    for (uint32_t head = 0; head < heads; ++head) {
        const size_t base = size_t(head) * seq * CubeFA::D;
        for (uint32_t row = 0; row < seq; ++row) {
            std::vector<double> scores(row + 1);
            double maximum = -INFINITY;
            for (uint32_t token = 0; token <= row; ++token) {
                double dot = 0.0;
                for (uint32_t d = 0; d < CubeFA::D; ++d)
                    dot += double(q[base + size_t(row) * CubeFA::D + d]) *
                           double(k[base + size_t(token) * CubeFA::D + d]);
                scores[token] = dot / std::sqrt(double(CubeFA::D));
                maximum = std::max(maximum, scores[token]);
            }
            double denominator = 0.0;
            std::vector<double> weighted(CubeFA::D, 0.0);
            for (uint32_t token = 0; token <= row; ++token) {
                const double weight = std::exp(scores[token] - maximum);
                denominator += weight;
                for (uint32_t d = 0; d < CubeFA::D; ++d)
                    weighted[d] += weight * double(v[base + size_t(token) * CubeFA::D + d]);
            }
            for (uint32_t d = 0; d < CubeFA::D; ++d) {
                const double expected = weighted[d] / denominator;
                const double actual = double(out[base + size_t(row) * CubeFA::D + d]);
                const double error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual) || error > 0.002 + 0.01 * std::abs(expected)) {
                    throw std::runtime_error("FP64 causal reference mismatch at head=" + std::to_string(head) +
                        " row=" + std::to_string(row) + " feature=" + std::to_string(d) +
                        " abs_error=" + std::to_string(error));
                }
            }
        }
    }

    std::printf("PASS BM=%u BN=%u PvRow32=%d BatchFix=%d PrioritizeP=%d DeferQkFixpipe=%d S=%u H=%u groups=%u repeat=%u KV_tiles=%llu Mmad=%llu max_abs=%.8g UB=%u\n",
        BM, BN, PvRow32, BatchFix, PrioritizeP, DeferQkFixpipe, seq, heads, groups, repeat, (unsigned long long)total_iterations,
        (unsigned long long)total_mmad_calls, max_error, G::UbBytes);
}

template<unsigned BM, unsigned BN>
void CheckShape() {
    for (unsigned seq : {128u, 256u})
        for (unsigned heads : {1u, 2u})
            for (unsigned groups : {1u, 3u})
                for (unsigned repeat : {0u, 1u})
                    Check<BM, BN>(seq, heads, groups, repeat);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--smoke") {
            Check<32, 64>(128, 1, 1, 0);
            std::puts("PASS mixed AIC/AIV CPU semantic smoke case");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--pv-row32") {
            Check<128, 512, true>(512, 2, 3, 0);
            Check<128, 512, true>(2560, 1, 3, 0);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--prioritize-p-writeback") {
            Check<128, 512, false, false, true>(512, 2, 3, 0);
            Check<128, 512, false, false, true>(1536, 1, 10, 0);
            Check<128, 512, false, false, true>(2560, 1, 3, 0);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--defer-qk-fixpipe") {
            Check<128, 512, false, false, false, true>(512, 2, 3, 0);
            Check<128, 512, false, false, false, true>(1536, 1, 10, 0);
            Check<128, 512, false, false, false, true>(2560, 1, 3, 0);
            return 0;
        }
        CheckShape<32, 64>();
        CheckShape<32, 128>();
        CheckShape<64, 64>();
        CheckShape<64, 128>();
        CheckShape<128, 128>();
        Check<128, 128>(512, 1, 3, 0);
        Check<32, 64>(640, 1, 3, 0);
        Check<128, 512>(512, 2, 3, 0);
        Check<128, 512>(2560, 1, 3, 0);
        Check<128, 512, true>(512, 2, 3, 0);
        Check<128, 512, true>(2560, 1, 3, 0);
        Check<128, 512, true, true>(512, 2, 3, 0);
        Check<128, 512, true, true>(2560, 1, 3, 0);
        Check<128, 512, false, false, true>(512, 2, 3, 0);
        Check<128, 512, false, false, true>(1536, 1, 10, 0);
        Check<128, 512, false, false, true>(2560, 1, 3, 0);
        Check<128, 512, false, false, false, true>(512, 2, 3, 0);
        Check<128, 512, false, false, false, true>(1536, 1, 10, 0);
        Check<128, 512, false, false, false, true>(2560, 1, 3, 0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("PASS all mixed AIC/AIV CPU semantic cases");
    return 0;
}
