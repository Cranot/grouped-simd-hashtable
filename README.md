# Grouped SIMD Hash Table

A high-performance C++ hash table that beats state-of-the-art at scale using grouped SIMD metadata scanning.

## Results

**vs ankerl::unordered_dense (Robin Hood, considered SOTA):**

| Size | ankerl | GroupedSIMD | Winner |
|------|--------|-------------|--------|
| 100k | 2.76 ms | 3.36 ms | ankerl |
| 500k | 29.2 ms | 34.0 ms | ankerl |
| **1M** | 72.3 ms | **71.9 ms** | **GroupedSIMD** |
| **2M** | 157.4 ms | **156.3 ms** | **GroupedSIMD** |

**Operation breakdown at 1M elements (80% load):**

| Operation | ankerl | GroupedSIMD | Speedup |
|-----------|--------|-------------|---------|
| Insert | 50.8 ms | 70.1 ms | 0.72x |
| Lookup Hit | 104 ms | 61.5 ms | **1.69x** |
| Lookup Miss | 5.4 ms | 4.5 ms | **1.21x** |

**Use this when:** Table size > 500k elements, lookup-heavy workloads. Insert overhead (0.72x) is acceptable when lookups dominate.

## Usage

```cpp
#include "grouped_simd_elastic.hpp"

// Create table with capacity for 1M elements
GroupedSIMDElastic<uint64_t, uint64_t> table(1200000);

// Insert
table.insert(key, value);

// Lookup
uint64_t* result = table.find(key);
if (result) {
    // Found: *result is the value
}

// Check existence
if (table.contains(key)) { ... }

// Subscript operator
table[key] = value;
```

## How It Works

### The Problem with SIMD in Hash Tables

Traditional quadratic probing accesses scattered memory locations:
```
Probe sequence: h, h+1, h+4, h+9, h+16, h+25...
```

To use SIMD, you'd need to **gather** from 16 random positions—slower than scalar code.

### The Solution: Grouped Probing

Probe 16 **contiguous** slots as a group, then jump to the next group:
```
Group 0: [h+0,  h+1,  ..., h+15]   ← SIMD scan (1 load)
Group 1: [h+16, h+17, ..., h+31]   ← SIMD scan (1 load)
Group 2: [h+32, h+33, ..., h+47]   ← SIMD scan (1 load)
```

Within each group, SSE2 scans all 16 metadata bytes in ~3 instructions:
```cpp
__m128i metadata = _mm_loadu_si128((__m128i*)&metadata_[base]);
__m128i matches  = _mm_cmpeq_epi8(metadata, target);
uint32_t mask    = _mm_movemask_epi8(matches);
```

This is the same insight behind Google's Swiss Tables.

### Metadata Format

Each slot has a 1-byte metadata tag:
- Bit 7: Occupied flag (1 = occupied, 0 = empty)
- Bits 0-6: 7-bit hash fragment

This filters out 127/128 non-matches before comparing keys.

## API Reference

```cpp
template <typename K, typename V, typename Hash = std::hash<K>>
class GroupedSIMDElastic {
    // Constructor: capacity and delta (1 - max_load_factor)
    explicit GroupedSIMDElastic(size_t capacity, double delta = 0.15);

    // Insert key-value pair. Returns false if table is full.
    bool insert(const K& key, const V& value);

    // Find value by key. Returns nullptr if not found.
    V* find(const K& key);
    const V* find(const K& key) const;

    // Check if key exists
    bool contains(const K& key) const;

    // Subscript operator (inserts default value if not found)
    V& operator[](const K& key);

    // Statistics
    size_t size() const;
    size_t capacity() const;
    double load_factor() const;
    size_t max_probe_used() const;
};
```

## Requirements

- C++17 or later
- SSE2 support (standard on all x86-64 CPUs)
- Header-only, no dependencies

## Benchmarking

```bash
# Compile
g++ -O3 -march=native -msse2 -std=c++17 -o benchmark benchmark_final_sota.cpp

# Run
./benchmark
```

## The Research Journey

This implementation emerged from exploring the February 2025 "Elastic Hashing" paper that disproved Yao's 40-year-old conjecture about uniform probing.

**What we tried:**

| Variant | Result | Learning |
|---------|--------|----------|
| Non-greedy probing | 5% faster at 1M | O(1) amortized works |
| SIMD (scattered) | 0.18x (5x slower) | Gather overhead kills SIMD |
| Memory prefetching | 0.41x (2.5x slower) | Hardware prefetcher already wins |
| Robin Hood | 2x faster miss | Low probe variance matters |
| **Grouped SIMD** | **1.5x faster lookup** | Contiguous access enables SIMD |

**Key insight:** SIMD requires contiguous memory access. Quadratic probing scatters accesses, defeating SIMD. Grouped probing (Swiss Tables' approach) solves this.

## Limitations & Trade-offs

| Trade-off | Impact | Notes |
|-----------|--------|-------|
| Insert overhead | 0.72x vs ankerl | Non-greedy candidate collection |
| Small tables | Loses below 500k | Crossover at ~500k-1M elements |
| No deletion | Not implemented | Planned for future |
| No resizing | Fixed capacity | Must pre-size |
| SSE2 only | x86-64 only | No ARM NEON version |

### Technical: Why Quadratic Group Jumps?

Groups use **quadratic jumps** to avoid clustering:
```
Group 0: h + 16×0² = h
Group 1: h + 16×1² = h+16
Group 2: h + 16×2² = h+64
Group 3: h + 16×3² = h+144
```

Linear jumps (h, h+16, h+32...) caused **42% insert failure rate** due to probe sequence overlap. Quadratic jumps spread groups across the table, ensuring all slots are reachable.

## Files

```
grouped_simd_elastic.hpp    # Main implementation (ship this)
hybrid_elastic.hpp          # Non-SIMD baseline
benchmark_final_sota.cpp    # Benchmark vs ankerl
INSIGHTS.md                 # Full research log
EXPERIMENT_RESULTS.md       # All experiment data
```

## License

MIT

## Acknowledgments

- [Swiss Tables](https://abseil.io/about/design/swisstables) (Google) - The grouped probing insight
- [ankerl::unordered_dense](https://github.com/martinus/unordered_dense) - SOTA benchmark baseline
- [Elastic Hashing paper](https://arxiv.org/abs/2501.02305) - Theoretical foundation

---

<p align="center">
  <b>Grouped SIMD Hash Table</b> - SOTA hashing at scale<br>
  Made by <a href="https://github.com/Cranot">Dimitris Mitsos</a> & <a href="https://agentskb.com">AgentsKB.com</a><br>
  Using <a href="https://github.com/Cranot/deep-research">Deep Research</a>
</p>
