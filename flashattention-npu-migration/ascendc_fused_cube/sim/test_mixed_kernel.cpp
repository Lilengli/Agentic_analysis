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
constexpr uint8_t kRelayGuard = 0x6d;
constexpr size_t kGuardBytes = 64;

template<int Core, unsigned BM, unsigned BN>
void RunCore(uint32_t group, uint32_t sub, CubeFASim::Stats& stats,
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR relay, GM_ADDR masks,
        uint32_t heads, uint32_t seq, uint32_t groups) {
    CubeFASim::SetContext(group, Core == AscendC::AIC ? CubeFASim::Core::AIC : CubeFASim::Core::AIV, sub, &stats);
    CubeFA::RunCore<Core, BM, BN>(q, k, v, out, relay, masks, heads, seq, groups);
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

template<unsigned BM, unsigned BN>
void CheckStats(uint32_t group, uint32_t groups, uint32_t heads, uint32_t seq,
        const CubeFASim::Stats& aic, const CubeFASim::Stats& av0,
        const CubeFASim::Stats& av1) {
    using G = CubeFA::Geometry<BM, BN>;
    constexpr uint32_t R = G::Rows;
    constexpr uint32_t MaxK = BN > CubeFA::D ? BN : CubeFA::D;
    constexpr uint32_t MaxN = MaxK;
    const uint64_t tasks = AssignedTasks(group, groups, uint64_t(heads) * (seq / BM));
    const uint64_t iterations = ExpectedIterations<BM, BN>(group, groups, heads, seq);
    std::array<uint64_t, CubeFA::ScoreSlots + 1> expected_pending{};
    std::array<uint64_t, CubeFA::ScoreSlots + 1> expected_pending_before_done{};
    std::vector<uint64_t> expected_prepared_at_pv_wait;
    const uint32_t query_tiles = seq / BM;
    uint64_t prepared = 0;
    for (uint64_t task = group; task < uint64_t(heads) * query_tiles; task += groups) {
        const uint32_t kv_tiles = uint32_t(task % query_tiles) * BM / BN + 1;
        for (uint32_t remaining = kv_tiles; remaining > 0; --remaining) {
            ++expected_pending[std::min(remaining, CubeFA::ScoreSlots)];
            ++expected_pending_before_done[std::min(remaining - 1, CubeFA::ScoreSlots)];
        }
        for (uint32_t tile = 0; tile < kv_tiles; ++tile)
            expected_prepared_at_pv_wait.push_back(prepared + std::min(tile + 2, kv_tiles));
        prepared += kv_tiles;
    }
    Require(aic.pending_qk_at_p_waits == expected_pending,
        "AIC did not fill the expected score slots before waiting for P");
    Require(aic.pending_qk_at_done_waits == expected_pending_before_done,
        "AIC did not refill the score ring before waiting for TILE_DONE");
    Require(aic.mmad_calls == 2 * iterations,
        "AIC group " + std::to_string(group) + " did not issue exactly two Mmad products per KV tile");
    Require(aic.gm_write_elements == iterations * (uint64_t(BM) * BN + uint64_t(BM) * CubeFA::D),
        "AIC score/delta workspace write count is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::A1)] ==
                uint64_t(BM) * CubeFA::D * 2 + uint64_t(BM) * BN * 2 + uint64_t(BN) * CubeFA::D * 2,
        "AIC A1 buffer sizes do not match the mixed schedule");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::A2)] == uint64_t(BM) * MaxK * 2,
        "AIC A2 buffer size is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::B2)] == uint64_t(BN) * CubeFA::D * 2,
        "AIC B2 buffer size is incorrect");
    Require(aic.local_bytes_by_position[unsigned(AscendC::TPosition::CO1)] == 2 * uint64_t(BM) * MaxN * 4,
        "AIC CO1 buffer size is incorrect");

    for (const CubeFASim::Stats* av : {&av0, &av1}) {
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
            Require(snapshot.gm_read_elements == mask_reads + (scores_prepared * BN + i * CubeFA::D) * R,
                "AIV did not prefetch the next score before waiting for PV");
            Require(snapshot.whole_reduce_max_calls == scores_prepared * (BN == 128 ? 2 : 1),
                "AIV did not reduce the next score before waiting for PV");
        }
        Require(av->gm_read_elements == mask_reads + iterations * (uint64_t(R) * BN + uint64_t(R) * CubeFA::D),
            "AIV mask/score/delta GM read count is incorrect");
        Require(av->gm_write_elements == iterations * uint64_t(R) * BN + tasks * uint64_t(R) * CubeFA::D,
            "AIV probability/output GM write count is incorrect");
        const uint64_t reduce_calls = iterations * (BN == 128 ? 2 : 1);
        Require(av->whole_reduce_max_calls == reduce_calls && av->whole_reduce_sum_calls == reduce_calls,
            "AIV reduction calls do not match the score tile width");
        Require(av->max_calls == iterations * (BN == 128 ? 2 : 1),
            "AIV must combine the split maxima and update the running maxima as vectors");
        Require(av->brcb_calls == iterations && av->sub_calls == iterations * (1 + BN / 64),
            "AIV must compute the alpha exponent and subtract broadcast maxima with vector operations");
        Require(av->mul_calls == iterations && av->exp_calls == 2 * iterations,
            "AIV online-softmax vector operation counts are incorrect");
        Require(av->cast_calls == iterations + tasks,
            "AIV probability or output conversion count is incorrect");
        Require(av->duplicate_calls == 3 * tasks && av->select_calls == tasks,
            "AIV must initialize online state and select the packed mask once per query tile");
    }
}

template<unsigned BM, unsigned BN>
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
                RunCore<AscendC::AIC, BM, BN>(group, 0, aic_stats[group],
                    reinterpret_cast<GM_ADDR>(q.data()), reinterpret_cast<GM_ADDR>(k.data()),
                    reinterpret_cast<GM_ADDR>(v.data()), reinterpret_cast<GM_ADDR>(out.data()),
                    relay_base, mask_base, heads, seq, groups);
            } catch (const std::exception& e) { record_error(group, e); }
        });
        for (uint32_t sub = 0; sub < 2; ++sub) {
            workers.emplace_back([&, group, sub] {
                try {
                    CubeFASim::Stats& stats = sub == 0 ? aiv0_stats[group] : aiv1_stats[group];
                    RunCore<AscendC::AIV, BM, BN>(group, sub, stats,
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
        CheckStats<BM, BN>(group, groups, heads, seq, aic_stats[group], aiv0_stats[group], aiv1_stats[group]);
        const uint64_t iterations = ExpectedIterations<BM, BN>(group, groups, heads, seq);
        total_iterations += iterations;
        total_mmad_calls += aic_stats[group].mmad_calls;
        total_output_writes += aiv0_stats[group].gm_write_elements - iterations * (BM / 2) * BN;
        total_output_writes += aiv1_stats[group].gm_write_elements - iterations * (BM / 2) * BN;
    }
    Require(total_mmad_calls == 2 * total_iterations, "total AIC Mmad count is not two products per KV tile");
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

    std::printf("PASS BM=%u BN=%u S=%u H=%u groups=%u repeat=%u KV_tiles=%llu Mmad=%llu max_abs=%.8g UB=%u\n",
        BM, BN, seq, heads, groups, repeat, (unsigned long long)total_iterations,
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
        CheckShape<32, 64>();
        CheckShape<32, 128>();
        CheckShape<64, 64>();
        CheckShape<64, 128>();
        CheckShape<128, 128>();
        Check<128, 128>(512, 1, 3, 0);
        Check<32, 64>(640, 1, 3, 0);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("PASS all mixed AIC/AIV CPU semantic cases");
    return 0;
}
