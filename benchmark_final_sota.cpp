/**
 * FINAL SOTA COMPARISON
 * =====================
 * GroupedSIMDElastic vs ankerl::unordered_dense (Robin Hood SOTA)
 */

#include "grouped_simd_elastic.hpp"
#include "hybrid_elastic.hpp"
#include "ankerl_unordered_dense.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

using namespace std;
using namespace std::chrono;

template <typename Func>
double time_ms(Func&& func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

int main() {
    cout << "============================================================\n";
    cout << "  FINAL SOTA COMPARISON: GroupedSIMD vs ankerl\n";
    cout << "============================================================\n\n";

    vector<size_t> sizes = {10000, 100000, 500000, 1000000, 2000000};

    cout << left << setw(10) << "Size"
         << right << setw(12) << "ankerl"
         << setw(12) << "Hybrid"
         << setw(12) << "GroupedSIMD"
         << setw(12) << "GS/ankerl"
         << setw(12) << "Verdict" << "\n";
    cout << string(70, '-') << "\n";

    for (size_t n : sizes) {
        // Generate test data
        mt19937_64 rng(42);
        vector<uint64_t> keys(n);
        for (size_t i = 0; i < n; ++i) {
            keys[i] = rng();
        }

        vector<uint64_t> lookup_keys(keys.begin(), keys.begin() + n/10);
        shuffle(lookup_keys.begin(), lookup_keys.end(), rng);

        vector<uint64_t> miss_keys(n/10);
        for (auto& k : miss_keys) k = rng();

        // === ankerl::unordered_dense ===
        double ankerl_insert, ankerl_hit, ankerl_miss;
        {
            ankerl::unordered_dense::map<uint64_t, uint64_t> map;
            map.reserve(n);

            ankerl_insert = time_ms([&]() {
                for (size_t i = 0; i < n; ++i) {
                    map[keys[i]] = i;
                }
            });

            volatile uint64_t sink = 0;
            ankerl_hit = time_ms([&]() {
                for (auto k : lookup_keys) {
                    auto it = map.find(k);
                    if (it != map.end()) sink += it->second;
                }
            });

            ankerl_miss = time_ms([&]() {
                for (auto k : miss_keys) {
                    auto it = map.find(k);
                    if (it != map.end()) sink += it->second;
                }
            });
        }

        // === HybridElastic (baseline) ===
        double hybrid_insert, hybrid_hit, hybrid_miss;
        size_t hybrid_maxprobe;
        {
            size_t capacity = static_cast<size_t>(n / 0.85);
            HybridElastic<uint64_t, uint64_t> table(capacity);

            hybrid_insert = time_ms([&]() {
                for (size_t i = 0; i < n; ++i) {
                    table.insert(keys[i], i);
                }
            });

            volatile uint64_t sink = 0;
            hybrid_hit = time_ms([&]() {
                for (auto k : lookup_keys) {
                    auto* p = table.find(k);
                    if (p) sink += *p;
                }
            });

            hybrid_miss = time_ms([&]() {
                for (auto k : miss_keys) {
                    auto* p = table.find(k);
                    if (p) sink += *p;
                }
            });

            hybrid_maxprobe = table.max_probe_used();
        }

        // === GroupedSIMDElastic (our new champion) ===
        double gs_insert, gs_hit, gs_miss;
        size_t gs_maxprobe;
        {
            size_t capacity = static_cast<size_t>(n / 0.85);
            GroupedSIMDElastic<uint64_t, uint64_t> table(capacity);

            gs_insert = time_ms([&]() {
                for (size_t i = 0; i < n; ++i) {
                    table.insert(keys[i], i);
                }
            });

            volatile uint64_t sink = 0;
            gs_hit = time_ms([&]() {
                for (auto k : lookup_keys) {
                    auto* p = table.find(k);
                    if (p) sink += *p;
                }
            });

            gs_miss = time_ms([&]() {
                for (auto k : miss_keys) {
                    auto* p = table.find(k);
                    if (p) sink += *p;
                }
            });

            gs_maxprobe = table.max_probe_used();
        }

        // Combined metric (harmonic mean of ops)
        double ankerl_total = ankerl_insert + ankerl_hit + ankerl_miss;
        double hybrid_total = hybrid_insert + hybrid_hit + hybrid_miss;
        double gs_total = gs_insert + gs_hit + gs_miss;

        double ratio = ankerl_total / gs_total;
        string verdict = (ratio > 1.0) ? "GS WINS" : (ratio > 0.9) ? "~TIE" : "ankerl";

        cout << left << setw(10) << n
             << right << setw(12) << fixed << setprecision(2) << ankerl_total
             << setw(12) << hybrid_total
             << setw(12) << gs_total
             << setw(12) << ratio << "x"
             << setw(12) << verdict << "\n";
    }

    cout << "\n============================================================\n";
    cout << "  DETAILED BREAKDOWN (1M elements)\n";
    cout << "============================================================\n\n";

    size_t n = 1000000;
    mt19937_64 rng(42);
    vector<uint64_t> keys(n);
    for (size_t i = 0; i < n; ++i) keys[i] = rng();

    vector<uint64_t> lookup_keys(keys.begin(), keys.begin() + n/10);
    shuffle(lookup_keys.begin(), lookup_keys.end(), rng);

    vector<uint64_t> miss_keys(n/10);
    for (auto& k : miss_keys) k = rng();

    // ankerl
    ankerl::unordered_dense::map<uint64_t, uint64_t> ankerl_map;
    ankerl_map.reserve(n);

    double a_ins = time_ms([&]() { for (size_t i = 0; i < n; ++i) ankerl_map[keys[i]] = i; });

    volatile uint64_t sink = 0;
    double a_hit = time_ms([&]() { for (auto k : lookup_keys) { auto it = ankerl_map.find(k); if (it != ankerl_map.end()) sink += it->second; } });
    double a_miss = time_ms([&]() { for (auto k : miss_keys) { auto it = ankerl_map.find(k); if (it != ankerl_map.end()) sink += it->second; } });

    // GroupedSIMD
    size_t capacity = static_cast<size_t>(n / 0.85);
    GroupedSIMDElastic<uint64_t, uint64_t> gs_table(capacity);

    double g_ins = time_ms([&]() { for (size_t i = 0; i < n; ++i) gs_table.insert(keys[i], i); });
    double g_hit = time_ms([&]() { for (auto k : lookup_keys) { auto* p = gs_table.find(k); if (p) sink += *p; } });
    double g_miss = time_ms([&]() { for (auto k : miss_keys) { auto* p = gs_table.find(k); if (p) sink += *p; } });

    cout << left << setw(15) << "Operation"
         << right << setw(12) << "ankerl(ms)"
         << setw(12) << "GS(ms)"
         << setw(12) << "Speedup" << "\n";
    cout << string(51, '-') << "\n";

    cout << left << setw(15) << "Insert"
         << right << setw(12) << fixed << setprecision(2) << a_ins
         << setw(12) << g_ins
         << setw(12) << a_ins/g_ins << "x\n";

    cout << left << setw(15) << "Lookup Hit"
         << right << setw(12) << a_hit
         << setw(12) << g_hit
         << setw(12) << a_hit/g_hit << "x\n";

    cout << left << setw(15) << "Lookup Miss"
         << right << setw(12) << a_miss
         << setw(12) << g_miss
         << setw(12) << a_miss/g_miss << "x\n";

    cout << "\nMaxProbe: GroupedSIMD = " << gs_table.max_probe_used() << "\n";

    cout << "\n============================================================\n";
    if (a_ins/g_ins > 1 && a_hit/g_hit > 1 && a_miss/g_miss > 1) {
        cout << "  RESULT: GroupedSIMD BEATS SOTA on ALL operations!\n";
    } else if ((a_ins + a_hit + a_miss) / (g_ins + g_hit + g_miss) > 1) {
        cout << "  RESULT: GroupedSIMD BEATS SOTA overall!\n";
    } else {
        cout << "  RESULT: ankerl (SOTA) still wins\n";
    }
    cout << "============================================================\n";

    return 0;
}
