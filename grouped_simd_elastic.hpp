/**
 * Grouped SIMD Elastic Hash Table
 * =================================
 *
 * Fixes the SIMD failure by using GROUPED probing instead of scattered quadratic probing.
 *
 * Key insight from Swiss Tables:
 * - Probe in GROUPS of 16 contiguous slots
 * - SIMD scan within each group (fast, contiguous)
 * - Jump quadratically BETWEEN groups (still good distribution)
 *
 * Probing pattern:
 * - Group 0: slots [h+0, h+1, h+2, ..., h+15]
 * - Group 1: slots [h+16, h+17, ..., h+31]        (offset = 16*1)
 * - Group 2: slots [h+32, h+33, ..., h+47]        (offset = 16*2, linear)
 * - Group j: slots [h + 16*j, ...]                (linear group jumps)
 *
 * Why this works:
 * 1. Within-group: contiguous metadata → SIMD _mm_loadu_si128 is FREE
 * 2. Between-groups: linear jumps are simple and safe (no overflow)
 * 3. No GATHER needed: metadata is naturally aligned in memory
 *
 * Previous SIMD attempt: 0.18x slower (scattered GATHER)
 * This version: should match or beat HybridElastic
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <functional>
#include <random>
#include <stdexcept>
#include <emmintrin.h>  // SSE2

#ifdef _MSC_VER
    #include <intrin.h>
#endif

template <typename K, typename V, typename Hash = std::hash<K>>
class GroupedSIMDElastic {
public:
    struct Entry {
        K key;
        V value;
    };

private:
    // Metadata: 7-bit hash fragment + 1-bit occupied
    // Empty = 0x00, Occupied = 0x80 | (hash >> 57)
    std::vector<uint8_t> metadata_;
    std::vector<Entry> table_;
    size_t capacity_;
    size_t size_ = 0;
    size_t max_inserts_;
    double delta_;
    size_t max_probe_limit_;
    size_t max_group_used_ = 0;  // Track groups, not individual probes
    uint64_t salt_;
    Hash hasher_;

    static constexpr double C = 4.0;
    static constexpr size_t GROUP_SIZE = 16;  // SSE2 processes 16 bytes
    static constexpr size_t EARLY_EXIT_GROUPS = 1;  // Greedy for first group
    static constexpr uint8_t EMPTY = 0x00;
    static constexpr uint8_t OCCUPIED_BIT = 0x80;

    uint64_t hash_with_salt(const K& key) const {
        return hasher_(key) ^ salt_;
    }

    uint8_t hash_fragment(uint64_t h) const {
        return static_cast<uint8_t>((h >> 57) & 0x7F);
    }

    uint8_t make_metadata(uint64_t h) const {
        return OCCUPIED_BIT | hash_fragment(h);
    }

    // Group base index: linear jump between groups (simpler, avoids overflow)
    // We sacrifice perfect quadratic distribution for safety and simplicity
    size_t group_base(uint64_t h, size_t group_idx) const {
        // Group j starts at: h + GROUP_SIZE * j
        return (h + GROUP_SIZE * group_idx) % capacity_;
    }

    // Get slot index within a group (handles wraparound)
    size_t slot_in_group(size_t base, size_t offset) const {
        return (base + offset) % capacity_;
    }

    // Count how many groups we need to check
    size_t max_groups() const {
        size_t max_g = (max_probe_limit_ + GROUP_SIZE - 1) / GROUP_SIZE;
        // Make sure we don't go beyond capacity
        size_t max_possible = (capacity_ + GROUP_SIZE - 1) / GROUP_SIZE;
        return (max_g < max_possible) ? max_g : max_possible;
    }

public:
    explicit GroupedSIMDElastic(size_t capacity, double delta = 0.1)
        : capacity_(capacity)
        , delta_(delta)
        , metadata_(capacity, EMPTY)
        , table_(capacity)
    {
        if (capacity == 0) throw std::invalid_argument("Capacity must be positive");
        if (delta <= 0 || delta >= 1) throw std::invalid_argument("Delta must be in (0,1)");

        max_inserts_ = capacity - static_cast<size_t>(delta * capacity);
        max_probe_limit_ = static_cast<size_t>(C * std::log2(1.0 / delta));
        if (max_probe_limit_ < GROUP_SIZE) max_probe_limit_ = GROUP_SIZE;
        if (max_probe_limit_ > capacity) max_probe_limit_ = capacity;

        std::random_device rd;
        salt_ = rd();
    }

    bool insert(const K& key, const V& value) {
        if (size_ >= max_inserts_) {
            return false;
        }

        uint64_t h = hash_with_salt(key);
        uint8_t meta = make_metadata(h);

        // === EARLY EXIT: Check first group greedily ===
        size_t base0 = group_base(h, 0);

        // Try to find contiguous group without wraparound first
        if (base0 + GROUP_SIZE <= capacity_) {
            // Fast path: group is contiguous, can use SIMD
            __m128i meta_vec = _mm_loadu_si128((__m128i*)&metadata_[base0]);
            __m128i empty_vec = _mm_set1_epi8(EMPTY);
            __m128i empty_cmp = _mm_cmpeq_epi8(meta_vec, empty_vec);
            int empty_mask = _mm_movemask_epi8(empty_cmp);

            // Check for existing key with same metadata
            __m128i target_vec = _mm_set1_epi8(meta);
            __m128i match_cmp = _mm_cmpeq_epi8(meta_vec, target_vec);
            int match_mask = _mm_movemask_epi8(match_cmp);

            // Process metadata matches (check keys)
            while (match_mask != 0) {
                unsigned long bit_idx;
                #ifdef _MSC_VER
                    _BitScanForward(&bit_idx, match_mask);
                #else
                    bit_idx = __builtin_ctz(match_mask);
                #endif

                size_t idx = base0 + bit_idx;
                if (table_[idx].key == key) {
                    table_[idx].value = value;
                    return true;
                }
                match_mask &= (match_mask - 1);
            }

            // Take first empty slot
            if (empty_mask != 0) {
                unsigned long bit_idx;
                #ifdef _MSC_VER
                    _BitScanForward(&bit_idx, empty_mask);
                #else
                    bit_idx = __builtin_ctz(empty_mask);
                #endif

                size_t idx = base0 + bit_idx;
                metadata_[idx] = meta;
                table_[idx] = {key, value};
                ++size_;
                if (0 > max_group_used_) max_group_used_ = 0;
                return true;
            }
        } else {
            // Wraparound case: fall back to scalar for first group
            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                size_t idx = slot_in_group(base0, i);

                if (metadata_[idx] == EMPTY) {
                    metadata_[idx] = meta;
                    table_[idx] = {key, value};
                    ++size_;
                    if (0 > max_group_used_) max_group_used_ = 0;
                    return true;
                }

                if (metadata_[idx] == meta && table_[idx].key == key) {
                    table_[idx].value = value;
                    return true;
                }
            }
        }

        // === NON-GREEDY: Collect candidates from remaining groups ===
        double load = static_cast<double>(size_) / capacity_;
        size_t max_groups_to_check = (load > 0.8) ? 8 : 4;
        size_t total_groups = max_groups();
        if (max_groups_to_check > total_groups) max_groups_to_check = total_groups;

        struct Candidate {
            size_t group_idx;
            size_t slot_offset;
            size_t table_idx;
        };
        Candidate candidates[128];
        size_t num_candidates = 0;

        for (size_t g = EARLY_EXIT_GROUPS; g < max_groups_to_check && num_candidates < 128; ++g) {
            size_t base = group_base(h, g);

            // Check if group is contiguous
            if (base + GROUP_SIZE <= capacity_) {
                // SIMD scan
                __m128i meta_vec = _mm_loadu_si128((__m128i*)&metadata_[base]);
                __m128i empty_vec = _mm_set1_epi8(EMPTY);
                __m128i empty_cmp = _mm_cmpeq_epi8(meta_vec, empty_vec);
                int empty_mask = _mm_movemask_epi8(empty_cmp);

                // Check for existing key
                __m128i target_vec = _mm_set1_epi8(meta);
                __m128i match_cmp = _mm_cmpeq_epi8(meta_vec, target_vec);
                int match_mask = _mm_movemask_epi8(match_cmp);

                while (match_mask != 0) {
                    unsigned long bit_idx;
                    #ifdef _MSC_VER
                        _BitScanForward(&bit_idx, match_mask);
                    #else
                        bit_idx = __builtin_ctz(match_mask);
                    #endif

                    size_t idx = base + bit_idx;
                    if (table_[idx].key == key) {
                        table_[idx].value = value;
                        return true;
                    }
                    match_mask &= (match_mask - 1);
                }

                // Collect empty slots
                while (empty_mask != 0) {
                    unsigned long bit_idx;
                    #ifdef _MSC_VER
                        _BitScanForward(&bit_idx, empty_mask);
                    #else
                        bit_idx = __builtin_ctz(empty_mask);
                    #endif

                    candidates[num_candidates++] = {g, bit_idx, base + bit_idx};
                    empty_mask &= (empty_mask - 1);
                }
            } else {
                // Wraparound: scalar scan
                for (size_t i = 0; i < GROUP_SIZE; ++i) {
                    size_t idx = slot_in_group(base, i);

                    if (metadata_[idx] == EMPTY) {
                        candidates[num_candidates++] = {g, i, idx};
                    } else if (metadata_[idx] == meta && table_[idx].key == key) {
                        table_[idx].value = value;
                        return true;
                    }
                }
            }
        }

        // Pick best candidate (earliest group, earliest slot in group)
        if (num_candidates > 0) {
            size_t best_i = 0;
            for (size_t i = 1; i < num_candidates; ++i) {
                if (candidates[i].group_idx < candidates[best_i].group_idx ||
                    (candidates[i].group_idx == candidates[best_i].group_idx &&
                     candidates[i].slot_offset < candidates[best_i].slot_offset)) {
                    best_i = i;
                }
            }

            size_t idx = candidates[best_i].table_idx;
            size_t grp = candidates[best_i].group_idx;

            metadata_[idx] = meta;
            table_[idx] = {key, value};
            ++size_;
            if (grp > max_group_used_) max_group_used_ = grp;
            return true;
        }

        // Fallback: scan all remaining groups
        for (size_t g = max_groups_to_check; g < total_groups; ++g) {
            size_t base = group_base(h, g);

            for (size_t i = 0; i < GROUP_SIZE; ++i) {
                size_t idx = slot_in_group(base, i);

                if (metadata_[idx] == EMPTY) {
                    metadata_[idx] = meta;
                    table_[idx] = {key, value};
                    ++size_;
                    if (g > max_group_used_) max_group_used_ = g;
                    return true;
                } else if (metadata_[idx] == meta && table_[idx].key == key) {
                    table_[idx].value = value;
                    return true;
                }
            }
        }

        return false;
    }

    V* find(const K& key) {
        uint64_t h = hash_with_salt(key);
        uint8_t meta = make_metadata(h);
        size_t groups_to_check = max_group_used_ + 1;

        for (size_t g = 0; g < groups_to_check; ++g) {
            size_t base = group_base(h, g);

            // Check if group is contiguous (no wraparound)
            if (base + GROUP_SIZE <= capacity_) {
                // SIMD path: load 16 contiguous metadata bytes
                __m128i meta_vec = _mm_loadu_si128((__m128i*)&metadata_[base]);

                // Check for empty (early exit)
                __m128i empty_vec = _mm_set1_epi8(EMPTY);
                __m128i empty_cmp = _mm_cmpeq_epi8(meta_vec, empty_vec);
                int empty_mask = _mm_movemask_epi8(empty_cmp);

                // Check for metadata match
                __m128i target_vec = _mm_set1_epi8(meta);
                __m128i match_cmp = _mm_cmpeq_epi8(meta_vec, target_vec);
                int match_mask = _mm_movemask_epi8(match_cmp);

                // Process matches
                while (match_mask != 0) {
                    unsigned long bit_idx;
                    #ifdef _MSC_VER
                        _BitScanForward(&bit_idx, match_mask);
                    #else
                        bit_idx = __builtin_ctz(match_mask);
                    #endif

                    size_t idx = base + bit_idx;
                    if (table_[idx].key == key) {
                        return &table_[idx].value;
                    }
                    match_mask &= (match_mask - 1);
                }

                // Early exit if we hit an empty slot
                if (empty_mask != 0) {
                    return nullptr;
                }
            } else {
                // Wraparound: fall back to scalar
                for (size_t i = 0; i < GROUP_SIZE; ++i) {
                    size_t idx = slot_in_group(base, i);
                    uint8_t m = metadata_[idx];

                    if (m == EMPTY) {
                        return nullptr;
                    }

                    if (m == meta && table_[idx].key == key) {
                        return &table_[idx].value;
                    }
                }
            }
        }

        return nullptr;
    }

    const V* find(const K& key) const {
        return const_cast<GroupedSIMDElastic*>(this)->find(key);
    }

    bool contains(const K& key) const {
        return find(key) != nullptr;
    }

    V& operator[](const K& key) {
        V* ptr = find(key);
        if (ptr) return *ptr;
        insert(key, V{});
        return *find(key);
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    double load_factor() const { return static_cast<double>(size_) / capacity_; }
    size_t max_group_used() const { return max_group_used_; }
    size_t max_probe_limit() const { return max_probe_limit_; }

    // For benchmarking comparison
    size_t max_probe_used() const {
        return max_group_used_ * GROUP_SIZE + GROUP_SIZE - 1;
    }
};
