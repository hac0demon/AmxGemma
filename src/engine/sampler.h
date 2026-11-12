#pragma once
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace ni {

inline uint32_t pcg_hash(uint32_t v) {
  uint32_t state = v * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

struct SamplerConfig {
  float temp = 0.0f;               // 0 = greedy
  float top_p = 0.95f;             // nucleus sampling threshold
  uint32_t top_k = 40;             // top-k candidates (capped at 64)
  uint32_t seed = 42;              // random seed
  float repetition_penalty = 1.15f; // GGUF / HF repetition penalty (1.0 = disabled)
  float presence_penalty = 0.0f;   // OpenAI presence penalty
  float frequency_penalty = 0.0f;  // OpenAI frequency penalty
  size_t penalty_window = 256;     // number of recent tokens to penalize
};

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& cfg = {}) : cfg_(cfg) {}

  void set_config(const SamplerConfig& cfg) { cfg_ = cfg; }
  const SamplerConfig& config() const { return cfg_; }

  int sample(const float* logits, size_t vocab_size, uint32_t step,
             const int* history = nullptr, size_t history_len = 0) const {
    if (vocab_size == 0) return 0;

    // Apply repetition / presence / frequency penalties in-place and restore after
    bool apply_penalties = (history != nullptr && history_len > 0) &&
                           (cfg_.repetition_penalty != 1.0f ||
                            cfg_.presence_penalty != 0.0f ||
                            cfg_.frequency_penalty != 0.0f);

    struct ModEntry {
      int id;
      float original_val;
    };
    std::vector<ModEntry> restored;
    float* mutable_logits = const_cast<float*>(logits);

    if (apply_penalties) {
      size_t win = std::min(history_len, cfg_.penalty_window);
      const int* start = history + (history_len - win);

      std::vector<std::pair<int, int>> counts;
      counts.reserve(win);
      for (size_t i = 0; i < win; ++i) {
        int tid = start[i];
        if (tid >= 0 && tid < int(vocab_size)) {
          bool found = false;
          for (auto& p : counts) {
            if (p.first == tid) {
              p.second++;
              found = true;
              break;
            }
          }
          if (!found) counts.push_back({tid, 1});
        }
      }

      restored.reserve(counts.size());
      for (const auto& [tid, cnt] : counts) {
        float orig = mutable_logits[tid];
        restored.push_back({tid, orig});

        float val = orig;
        if (cfg_.repetition_penalty != 1.0f && cfg_.repetition_penalty > 0.0f) {
          if (val > 0.0f) val /= cfg_.repetition_penalty;
          else val *= cfg_.repetition_penalty;
        }
        val -= (cfg_.frequency_penalty * float(cnt) + cfg_.presence_penalty);
        mutable_logits[tid] = val;
      }
    }

    int chosen = 0;

    // Greedy fast path (over penalized logits)
    if (cfg_.temp <= 0.0f || cfg_.top_k <= 1) {
      float max_val = mutable_logits[0];
      vDSP_Length max_idx = 0;
      vDSP_maxvi(mutable_logits, 1, &max_val, &max_idx, vocab_size);
      chosen = int(max_idx);
    } else {
      uint32_t K = std::max(1u, std::min(cfg_.top_k, 64u));
      struct Cand {
        float val;
        int id;
        bool operator>(const Cand& o) const {
          return val > o.val || (val == o.val && id < o.id);
        }
      };

      // Min-heap to find top K candidates
      std::vector<Cand> heap;
      heap.reserve(K + 1);

      for (size_t i = 0; i < vocab_size; ++i) {
        float v = mutable_logits[i];
        if (std::isnan(v) || std::isinf(v)) continue;
        if (heap.size() < K) {
          heap.push_back({v, int(i)});
          if (heap.size() == K) {
            std::make_heap(heap.begin(), heap.end(), std::greater<Cand>());
          }
        } else if (v > heap.front().val) {
          std::pop_heap(heap.begin(), heap.end(), std::greater<Cand>());
          heap.back() = {v, int(i)};
          std::push_heap(heap.begin(), heap.end(), std::greater<Cand>());
        }
      }

      if (heap.empty()) {
        chosen = 0;
      } else {
        std::sort(heap.begin(), heap.end(), [](const Cand& a, const Cand& b) {
          return a.val > b.val || (a.val == b.val && a.id < b.id);
        });

        uint32_t actual_k = uint32_t(heap.size());
        float max_v = heap[0].val;

        // Top-P filter on untempered distribution
        std::vector<float> probs(actual_k);
        float tot = 0.0f;
        for (uint32_t r = 0; r < actual_k; ++r) {
          probs[r] = std::exp(heap[r].val - max_v);
          tot += probs[r];
        }

        uint32_t keep = actual_k;
        float cum = 0.0f;
        for (uint32_t r = 0; r < actual_k; ++r) {
          cum += probs[r] / tot;
          if (cum >= cfg_.top_p) {
            keep = r + 1;
            break;
          }
        }

        // Tempered distribution over surviving candidates
        float kt = 0.0f;
        for (uint32_t r = 0; r < keep; ++r) {
          probs[r] = std::exp((heap[r].val - max_v) / cfg_.temp);
          kt += probs[r];
        }

        uint32_t hash_val = pcg_hash(cfg_.seed ^ pcg_hash(step));
        float u = float(hash_val & 0xffffff) / 16777216.0f * kt;

        chosen = heap[keep - 1].id;
        float acc = 0.0f;
        for (uint32_t r = 0; r < keep; ++r) {
          acc += probs[r];
          if (u < acc) {
            chosen = heap[r].id;
            break;
          }
        }
      }
    }

    // Restore modified logits
    for (const auto& entry : restored) {
      mutable_logits[entry.id] = entry.original_val;
    }

    return chosen;
  }

 private:
  SamplerConfig cfg_;
};

} // namespace ni
