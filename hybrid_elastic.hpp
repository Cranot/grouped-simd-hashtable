/**
 * Hybrid Elastic Hash Table
 * ==========================
 *
 * Combines the best of:
 * - Swiss Tables: 1-byte metadata for fast filtering
 * - NonGreedy: bounded probes for O(1) amortized lookup
 * - Adaptive: greedy when sparse, non-greedy when dense
 *
 * Key optimizations:
 * 1. Early-exit: if first 4 probes have an empty, take it
 * 2. Metadata filtering: 7-bit hash + occupied flag
 * 3. Adaptive non-greedy based on load factor
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <optional>
#include <functional>
#include <random>
#include <algorithm>
#include <stdexcept>

template <typename K, typename V, typename Hash = std::hash<K>>
class HybridElastic {
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
    size_t max_probe_used_ = 0;
    uint64_t salt_;
    Hash hasher_;

    static constexpr double C = 4.0;
    static constexpr size_t EARLY_EXIT_PROBES = 4;  // Greedy for first N probes
    static constexpr uint8_t EMPTY = 0x00;
    static constexpr uint8_t OCCUPIED_BIT = 0x80;

    uint64_t hash_with_salt(const K& key) const {
        return hasher_(key) ^ salt_;
    }

    // Extract 7-bit hash fragment for metadata
    uint8_t hash_fragment(uint64_t h) const {
        return static_cast<uint8_t>((h >> 57) & 0x7F);
    }

    uint8_t make_metadata(uint64_t h) const {
        return OCCUPIED_BIT | hash_fragment(h);
    }

    size_t probe_index(uint64_t h, size_t j) const {
        // Quadratic probing
        return (h + j * j) % capacity_;
    }

public:
    explicit HybridElastic(size_t capacity, double delta = 0.1)
        : capacity_(capacity)
        , delta_(delta)
        , metadata_(capacity, EMPTY)
        , table_(capacity)
    {
        if (capacity == 0) throw std::invalid_argument("Capacity must be positive");
        if (delta <= 0 || delta >= 1) throw std::invalid_argument("Delta must be in (0,1)");

        max_inserts_ = capacity - static_cast<size_t>(delta * capacity);
        max_probe_limit_ = static_cast<size_t>(C * std::log2(1.0 / delta));
        if (max_probe_limit_ < EARLY_EXIT_PROBES) max_probe_limit_ = EARLY_EXIT_PROBES;
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

        // === EARLY EXIT: Check first few probes for empty slot ===
        for (size_t j = 0; j < EARLY_EXIT_PROBES && j < capacity_; ++j) {
            size_t idx = probe_index(h, j);

            if (metadata_[idx] == EMPTY) {
                // Empty slot - take it immediately (greedy)
                metadata_[idx] = meta;
                table_[idx] = {key, value};
                ++size_;
                if (j > max_probe_used_) max_probe_used_ = j;
                return true;
            }

            // Check for existing key (metadata match first)
            if (metadata_[idx] == meta && table_[idx].key == key) {
                table_[idx].value = value;
                return true;
            }
        }

        // === NON-GREEDY: Collect candidates and pick best ===
        double load = static_cast<double>(size_) / capacity_;

        // Adaptive: more candidates at higher load
        size_t max_candidates = (load > 0.8) ? 16 : 8;

        struct Candidate {
            size_t probe_idx;
            size_t table_idx;
        };
        Candidate candidates[16];
        size_t num_candidates = 0;

        for (size_t j = EARLY_EXIT_PROBES; j < max_probe_limit_ && num_candidates < max_candidates; ++j) {
            size_t idx = probe_index(h, j);

            if (metadata_[idx] == EMPTY) {
                candidates[num_candidates++] = {j, idx};
            } else if (metadata_[idx] == meta && table_[idx].key == key) {
                table_[idx].value = value;
                return true;
            }
        }

        if (num_candidates > 0) {
            // Pick slot with lowest probe index
            size_t best_i = 0;
            for (size_t i = 1; i < num_candidates; ++i) {
                if (candidates[i].probe_idx < candidates[best_i].probe_idx) {
                    best_i = i;
                }
            }

            size_t best_idx = candidates[best_i].table_idx;
            size_t best_probe = candidates[best_i].probe_idx;

            metadata_[best_idx] = meta;
            table_[best_idx] = {key, value};
            ++size_;
            if (best_probe > max_probe_used_) max_probe_used_ = best_probe;
            return true;
        }

        // Fallback: scan remaining slots
        for (size_t j = max_probe_limit_; j < capacity_; ++j) {
            size_t idx = probe_index(h, j);

            if (metadata_[idx] == EMPTY) {
                metadata_[idx] = meta;
                table_[idx] = {key, value};
                ++size_;
                if (j > max_probe_used_) max_probe_used_ = j;
                return true;
            } else if (metadata_[idx] == meta && table_[idx].key == key) {
                table_[idx].value = value;
                return true;
            }
        }

        return false;
    }

    V* find(const K& key) {
        uint64_t h = hash_with_salt(key);
        uint8_t meta = make_metadata(h);
        size_t limit = max_probe_used_ + 1;

        for (size_t j = 0; j < limit; ++j) {
            size_t idx = probe_index(h, j);
            uint8_t m = metadata_[idx];

            // Empty = key not present in this chain
            if (m == EMPTY) {
                return nullptr;
            }

            // Metadata match = potential hit, verify key
            if (m == meta && table_[idx].key == key) {
                return &table_[idx].value;
            }
            // Metadata mismatch = definitely not this slot, continue
        }

        return nullptr;
    }

    const V* find(const K& key) const {
        return const_cast<HybridElastic*>(this)->find(key);
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
    size_t max_probe_used() const { return max_probe_used_; }
    size_t max_probe_limit() const { return max_probe_limit_; }
};
