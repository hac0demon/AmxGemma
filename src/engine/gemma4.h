#pragma once
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "util/bf16.h"

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ni {

#pragma pack(push, 1)
struct block_q8_0 {
  uint16_t d;     // FP16 scale
  int8_t qs[32];  // 32 quantized int8 values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

struct block_q4_0 {
  uint16_t d;     // FP16 scale
  uint8_t qs[16]; // 32 quantized 4-bit values (low nibble then high nibble)
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");
#pragma pack(pop)

inline uint16_t float_to_half(float x) {
  _Float16 f = (_Float16)x;
  uint16_t h;
  std::memcpy(&h, &f, 2);
  return h;
}

inline float half_to_float(uint16_t h) {
  _Float16 f;
  std::memcpy(&f, &h, 2);
  return float(f);
}

struct YarnConfig {
  bool enabled = false;
  uint32_t orig_ctx = 131072;          // native context length of Gemma 4 E2B (128K)
  uint32_t target_ctx = 262144;        // target extended context length (256K)
  float scale = 1.0f;                  // target_ctx / orig_ctx (e.g. 2.0f)
  float beta_fast = 32.0f;             // high-frequency boundary
  float beta_slow = 1.0f;              // low-frequency boundary
  float mscale = 1.0f;                 // attention temperature scaling factor
  float ext_factor = 1.0f;             // extrapolation blend factor (1.0 = YaRN ramp)

  // Precomputed values for fast inference
  float corr_dim_low = 0.0f;
  float corr_dim_high = 0.0f;
  float eff_mscale = 1.0f;

  void update(uint32_t head_dim, float freq_base) {
    if (!enabled || scale <= 1.0f) {
      eff_mscale = mscale;
      corr_dim_low = 0.0f;
      corr_dim_high = 0.0f;
      return;
    }
    float log_base_2 = 2.0f * std::log(freq_base);
    auto corr_dim = [&](float n_rot) -> float {
      return float(head_dim) * std::log(float(orig_ctx) / (n_rot * 2.0f * float(M_PI))) / log_base_2;
    };
    float start = std::floor(corr_dim(beta_fast));
    float end   = std::ceil(corr_dim(beta_slow));
    corr_dim_low = std::max(0.0f, start);
    corr_dim_high = std::min(float(head_dim / 2 - 1), end);
    eff_mscale = mscale;
    if (ext_factor != 0.0f) {
      eff_mscale *= (1.0f + 0.1f * std::log(scale));
    }
  }
};

struct Gemma4Config {
  uint32_t dim = 1536;                   // embedding_length (H)
  uint32_t n_layers = 35;                // block_count (L)
  uint32_t ffn_dim = 6144;               // feed_forward_length (I)
  uint32_t n_heads = 8;                  // head_count (H_q)
  uint32_t n_kv_heads = 1;               // head_count_kv (H_kv)
  uint32_t head_dim = 256;               // key_length (D_h)
  uint32_t head_dim_swa = 256;           // key_length_swa
  uint32_t sliding_window = 512;         // sliding_window (W_swa)
  uint32_t n_shared_kv = 0;              // shared_kv_layers (N_shared)
  uint32_t ple_dim = 256;                // embedding_length_per_layer_input (D_ple)
  uint32_t vocab_size = 262144;          // vocabulary size (V)
  float rope_freq_base = 1000000.0f;     // rope.freq_base (full attention)
  float rope_freq_base_swa = 10000.0f;   // rope.freq_base_swa (sliding window)
  float rms_norm_eps = 1e-6f;            // layer_norm_rms_epsilon
  float logit_softcap = 30.0f;           // final_logit_softcapping
  uint32_t max_context = 8192;           // maximum context length
  uint32_t prefill_step = 256;           // chunked prefill step size
  std::string kv_type_k = "q8_0";        // k-cache type (default q8_0)
  std::string kv_type_v = "q4_0";        // v-cache type (default q4_0)
  bool flash_attn = true;                // Flash Attention with online softmax
  YarnConfig yarn;                       // YaRN context extension configuration
};

struct LayerWeights {
  const gguf::Tensor* attn_norm = nullptr;
  const gguf::Tensor* attn_q = nullptr;
  const gguf::Tensor* attn_q_norm = nullptr;
  const gguf::Tensor* attn_k = nullptr;
  const gguf::Tensor* attn_k_norm = nullptr;
  const gguf::Tensor* attn_v = nullptr;
  const gguf::Tensor* attn_out = nullptr;
  const gguf::Tensor* post_attn_norm = nullptr;

  const gguf::Tensor* ffn_norm = nullptr;
  const gguf::Tensor* ffn_gate = nullptr;
  const gguf::Tensor* ffn_up = nullptr;
  const gguf::Tensor* ffn_down = nullptr;
  const gguf::Tensor* post_ffw_norm = nullptr;

  const gguf::Tensor* inp_gate = nullptr;
  const gguf::Tensor* ple_proj = nullptr;
  const gguf::Tensor* post_ple_norm = nullptr;

  const gguf::Tensor* layer_scale = nullptr;

  bool is_swa = true;
  bool has_kv = true;
  int kv_src_layer = -1;  // for shared KV layers

  uint32_t q_dim = 2048;   // lw.attn_q->rows()
  uint32_t kv_dim = 256;   // lw.attn_k->rows()
  uint32_t head_dim = 256; // q_dim / n_heads
  uint32_t n_heads = 8;
  uint32_t n_kv_heads = 1;
  uint32_t ffn_dim = 6144; // lw.ffn_gate->rows()

  // Byte offsets into contiguous kv_arena_
  size_t k_offset = 0;
  size_t v_offset = 0;
  size_t k_token_bytes = 0; // bytes per token for K cache (q8_0)
  size_t v_token_bytes = 0; // bytes per token for V cache (q4_0)
  size_t kv_capacity = 0;   // in tokens
};

struct KVSnapshot {
  std::vector<uint8_t> arena_bytes; // exact copy of kv_arena_ bytes
  std::vector<int32_t> primer_ids;  // baked token IDs
  int32_t seq_pos = 0;
  bool is_primed = false;
};

// ---------------------------------------------------------------------------
// QLoRA / LoRA Adapter Structures
// ---------------------------------------------------------------------------

struct LoraTensor {
  int r = 0;              // rank (e.g. 8, 16, 32)
  int d_in = 0;           // input dimension
  int d_out = 0;          // output dimension
  float alpha = 1.0f;     // lora scaling alpha
  float scale = 1.0f;     // alpha / r
  std::vector<float> a;   // [r x d_in] row-major
  std::vector<float> b;   // [d_out x r] row-major
};

struct LoraLayer {
  std::unique_ptr<LoraTensor> attn_q;
  std::unique_ptr<LoraTensor> attn_k;
  std::unique_ptr<LoraTensor> attn_v;
  std::unique_ptr<LoraTensor> attn_out;
  std::unique_ptr<LoraTensor> ffn_gate;
  std::unique_ptr<LoraTensor> ffn_up;
  std::unique_ptr<LoraTensor> ffn_down;
};

struct LoraAdapter {
  std::string id;                  // e.g. "logs_reducer", "log_compactor"
  float default_alpha = 16.0f;
  std::vector<LoraLayer> layers;    // size L = 35
};

// ---------------------------------------------------------------------------
// User Slot (Simultaneous multi-user access with independent KV arenas)
// ---------------------------------------------------------------------------

struct UserSlot {
  uint32_t slot_id = 0;
  std::string bound_user;          // user identifier string (from OpenAI API "user")
  int64_t last_accessed_ms = 0;
  std::vector<uint8_t> kv_arena;   // contiguous KV arena for this slot (bytes)
  int32_t seq_pos = 0;             // current sequence position in slot
  KVSnapshot snapshot;             // snapshot state for fast restore in this slot
  std::string attached_adapter_id; // attached LoraAdapter ID or ""
  bool is_active = false;
};

struct CompactPLE {
  void* mapped_ptr = nullptr;
  size_t mapped_size = 0;
  const uint16_t* codebooks = nullptr;
  const uint8_t* codes_packed = nullptr;
  uint32_t num_layers = 35;
  uint32_t num_groups = 32;
  uint32_t group_size = 8;
  uint32_t num_codes = 128;
  uint32_t ple_dim = 256;
  uint32_t vocab_size = 262144;
  float embed_scale = 16.0f;
  bool is_loaded = false;
};

class Gemma4Engine {
 public:
  Gemma4Engine();
  ~Gemma4Engine();

  // Load model weights and architecture parameters from GGUF file
  bool load_model(const gguf::File& file, uint32_t max_context = 8192, bool preload_amx = false);

  // Load compact AQLM PLE table from ple_{s,m,l}.safetensors
  bool load_ple_safetensors(const std::string& path);

  // -------------------------------------------------------------------------
  // Slot Management (Simultaneous access for >= 4 users)
  // -------------------------------------------------------------------------
  bool init_slots(uint32_t num_slots = 4);
  uint32_t num_slots() const { return uint32_t(slots_.size()); }
  UserSlot& slot(uint32_t idx) { return slots_.at(idx); }
  const UserSlot& slot(uint32_t idx) const { return slots_.at(idx); }

  // -------------------------------------------------------------------------
  // LoRA / QLoRA Adapter Management
  // -------------------------------------------------------------------------
  bool load_lora_adapter(const std::string& id, const std::string& path, float default_alpha = 16.0f);
  bool register_lora_adapter(std::unique_ptr<LoraAdapter> adapter);
  bool attach_adapter_to_slot(uint32_t slot_idx, const std::string& adapter_id);
  const LoraAdapter* get_adapter(const std::string& id) const;
  std::vector<std::string> list_adapters() const;

  // Clear active KV cache sequence in slot
  void reset_state(uint32_t slot_idx = 0);

  // Primer run for static system prompt -> fast snapshot in slot (or default slot 0)
  bool prime_system_prompt(const std::vector<int32_t>& system_tokens, uint32_t slot_idx = 0);

  // Fast restore snapshot (< 0.5 ms memcpy) into slot
  bool restore_snapshot(uint32_t slot_idx = 0);

  // Prefill a prompt for a specific slot (chunked in 256 tokens)
  // If keep_logits is true, logits for the very last token will be computed and returned
  const float* prefill(uint32_t slot_idx, const std::vector<int32_t>& tokens, bool keep_logits = true);
  const float* prefill(const std::vector<int32_t>& tokens, bool keep_logits = true) {
    return prefill(0, tokens, keep_logits);
  }

  // Decode a single token step (T = 1) for a specific slot
  const float* decode(uint32_t slot_idx, int32_t token);
  const float* decode(int32_t token) {
    return decode(0, token);
  }

  const Gemma4Config& config() const { return cfg_; }
  void set_yarn_config(const YarnConfig& yarn) {
    cfg_.yarn = yarn;
    cfg_.yarn.update(cfg_.head_dim, cfg_.rope_freq_base);
  }
  const YarnConfig& yarn_config() const { return cfg_.yarn; }

  // Rotary Position Embedding kernels (NEOX split-halves layout with proportional RoPE support)
  static void apply_rope(float* qk, size_t head_dim, int32_t pos, float freq_base, const float* rope_freqs = nullptr);
  static void apply_rope_yarn(float* qk, size_t head_dim, int32_t pos, float freq_base, const YarnConfig& yarn, bool use_yarn = true, const float* rope_freqs = nullptr);

  // Model Lifecycle & Resource Management (RAM savings on idle)
  void set_model_paths(const std::string& model_path, const std::string& ple_path = "") {
    model_path_ = model_path;
    ple_path_ = ple_path;
  }
  void unload_model();
  void unload_kv();
  bool is_model_loaded() const { return model_loaded_; }
  bool is_kv_loaded() const { return kv_loaded_; }
  void ensure_model_loaded();
  void ensure_kv_loaded(uint32_t slot_idx = 0);

  void set_flash_attn(bool enabled) { cfg_.flash_attn = enabled; }
  bool flash_attn_enabled() const { return cfg_.flash_attn; }

  // Flash Attention Kernel on CPU (online softmax)
  static void flash_attn_head(const float* q_h, const uint8_t* k_cache, const uint8_t* v_cache,
                              size_t k_token_bytes, size_t v_token_bytes, size_t cap,
                              bool is_swa, int32_t min_p, size_t valid_ctx, size_t hkv,
                              size_t Dh, float* out_h);

  // Quantization kernels (k: q8_0, v: q4_0)
  static void quantize_q8_0(const float* x, block_q8_0* y, size_t k);
  static void quantize_q4_0(const float* x, block_q4_0* y, size_t k);

  int32_t seq_pos(uint32_t slot_idx = 0) const {
    return slot_idx < slots_.size() ? slots_[slot_idx].seq_pos : seq_pos_;
  }
  const KVSnapshot& snapshot(uint32_t slot_idx = 0) const {
    return slot_idx < slots_.size() ? slots_[slot_idx].snapshot : snapshot_;
  }
  bool is_primed(uint32_t slot_idx = 0) const {
    return slot_idx < slots_.size() ? slots_[slot_idx].snapshot.is_primed : snapshot_.is_primed;
  }

 private:
  // Internal forward pass on a batch of T tokens (T <= prefill_step) for active slot
  void forward_chunk(uint32_t slot_idx, const int32_t* tokens, size_t n_tokens, int32_t start_pos, bool compute_logits);

  // Matmul helper (dispatches to Apple AMX cblas_sgemm or thread-parallel dot products)
  void matmul(const float* x, const gguf::Tensor* w, float* y, size_t M, size_t N, size_t K);

  // Apply LoRA low-rank delta Y += scale * (X * A^T) * B^T
  void apply_lora(const float* x, const LoraTensor& lora, float* y, size_t T);

  // On-the-fly row gather-dequantize
  void lookup_token_embd(const int32_t* tokens, size_t n_tokens, float* out);
  void lookup_ple_embd(const int32_t* tokens, size_t n_tokens, float* out);

  Gemma4Config cfg_;
  std::vector<LayerWeights> layers_;

  // Global weights
  const gguf::Tensor* token_embd_ = nullptr;
  const gguf::Tensor* output_norm_ = nullptr;
  const gguf::Tensor* output_weight_ = nullptr;
  const gguf::Tensor* rope_freqs_ = nullptr;

  // Global PLE weights
  const gguf::Tensor* ple_token_embd_ = nullptr;
  const gguf::Tensor* ple_model_proj_ = nullptr;
  const gguf::Tensor* ple_proj_norm_ = nullptr;

  CompactPLE compact_ple_;

  // Pre-converted FP32 weights if --preload-amx is enabled
  std::vector<std::vector<float>> preloaded_weights_;
  std::vector<const float*> tensor_preloaded_ptr_;

  // Contiguous KV cache arena (single flat vector of quantized bytes)
  std::vector<uint8_t> kv_arena_;
  int32_t seq_pos_ = 0;

  // Static snapshot of kv_arena_ for sub-millisecond restore
  KVSnapshot snapshot_;

  // Multi-user slots with dedicated KV arenas (>= 4 slots)
  std::vector<UserSlot> slots_;

  // Loaded LoRA / QLoRA adapters
  std::vector<std::unique_ptr<LoraAdapter>> adapters_;

  // Static activation arena (pre-allocated to avoid malloc in hot path)
  std::vector<float> act_arena_;
  std::vector<float> scores_buf_;
  std::vector<float> last_logits_;

  // Lifecycle state
  bool model_loaded_ = false;
  bool kv_loaded_ = false;
  std::string model_path_;
  std::string ple_path_;
  uint32_t max_context_ = 8192;
  bool preload_amx_ = false;
  std::unique_ptr<gguf::File> owned_file_;
  std::vector<int32_t> primer_tokens_;
};

} // namespace ni
