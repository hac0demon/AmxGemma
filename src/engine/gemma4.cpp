#include "engine/gemma4.h"
#include "util/log.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <string_view>
#include <nlohmann/json.hpp>

namespace ni {

using gguf::dequantize;

namespace {

// In-place RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * w
inline void rms_norm(const float* x, const float* w, float* y, size_t dim, float eps) {
  float sum_sq = 0.0f;
  vDSP_svesq(x, 1, &sum_sq, dim);
  float rms = std::sqrt(sum_sq / float(dim) + eps);
  float scale = 1.0f / rms;
  if (w) {
    for (size_t i = 0; i < dim; ++i) {
      y[i] = (x[i] * scale) * w[i];
    }
  } else {
    vDSP_vsmul(x, 1, &scale, y, 1, dim);
  }
}

// In-place GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_tanh(float x) {
  constexpr float k0 = 0.7978845608f; // sqrt(2 / pi)
  constexpr float k1 = 0.044715f;
  return 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x * x * x)));
}

} // namespace

// Neox RoPE on single head (split halves layout matching HuggingFace rotate_half & GGML_ROPE_TYPE_NEOX)
void Gemma4Engine::apply_rope(float* qk, size_t head_dim, int32_t pos, float freq_base, const float* rope_freqs) {
  size_t half = head_dim / 2;
  for (size_t i = 0; i < half; ++i) {
    if (rope_freqs && rope_freqs[i] > 1e10f) {
      // Unrotated dimension (proportional RoPE nope_angles)
      continue;
    }
    float theta = std::pow(freq_base, -float(2 * i) / float(head_dim));
    float angle = float(pos) * theta;
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    float x0 = qk[i];
    float x1 = qk[i + half];
    qk[i]        = x0 * cos_a - x1 * sin_a;
    qk[i + half] = x0 * sin_a + x1 * cos_a;
  }
}

// YaRN RoPE on single head (targeted to global full attention layers)
void Gemma4Engine::apply_rope_yarn(float* qk, size_t head_dim, int32_t pos, float freq_base, const YarnConfig& yarn, bool use_yarn, const float* rope_freqs) {
  if (!use_yarn || !yarn.enabled || yarn.scale <= 1.0f) {
    apply_rope(qk, head_dim, pos, freq_base, rope_freqs);
    return;
  }

  size_t half = head_dim / 2;
  float freq_scale = 1.0f / yarn.scale;
  float low = yarn.corr_dim_low;
  float high = yarn.corr_dim_high;
  float denom = std::max(0.001f, high - low);
  float mscale = yarn.eff_mscale;

  for (size_t i = 0; i < half; ++i) {
    if (rope_freqs && rope_freqs[i] > 1e10f) {
      // Unrotated dimension (proportional RoPE nope_angles)
      continue;
    }
    float theta_extrap = float(pos) * std::pow(freq_base, -float(2 * i) / float(head_dim));
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (yarn.ext_factor != 0.0f) {
      float y = (float(i) - low) / denom;
      float ramp_mix = (1.0f - std::min(1.0f, std::max(0.0f, y))) * yarn.ext_factor;
      theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }

    float cos_a = std::cos(theta) * mscale;
    float sin_a = std::sin(theta) * mscale;

    float x0 = qk[i];
    float x1 = qk[i + half];
    qk[i]        = x0 * cos_a - x1 * sin_a;
    qk[i + half] = x0 * sin_a + x1 * cos_a;
  }
}

void Gemma4Engine::quantize_q8_0(const float* x, block_q8_0* y, size_t k) {
  size_t nb = k / 32;
  for (size_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    const float* src = x + i * 32;
    for (int j = 0; j < 32; ++j) {
      float v = std::abs(src[j]);
      if (v > amax) amax = v;
    }
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    y[i].d = float_to_half(d);
    for (int j = 0; j < 32; ++j) {
      y[i].qs[j] = int8_t(std::round(src[j] * id));
    }
  }
}

void Gemma4Engine::quantize_q4_0(const float* x, block_q4_0* y, size_t k) {
  size_t nb = k / 32;
  for (size_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    float max_val = 0.0f;
    const float* src = x + i * 32;
    for (int j = 0; j < 32; ++j) {
      float v = src[j];
      if (std::abs(v) > amax) {
        amax = std::abs(v);
        max_val = v;
      }
    }
    const float d = max_val / -8.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    y[i].d = float_to_half(d);
    for (int j = 0; j < 16; ++j) {
      float x0 = src[j] * id;
      float x1 = src[16 + j] * id;
      uint8_t xi0 = uint8_t(std::min(15, std::max(0, int(x0 + 8.5f))));
      uint8_t xi1 = uint8_t(std::min(15, std::max(0, int(x1 + 8.5f))));
      y[i].qs[j] = xi0 | (xi1 << 4);
    }
  }
}

void Gemma4Engine::flash_attn_head(
    const float* q_h, const uint8_t* k_cache, const uint8_t* v_cache,
    size_t k_token_bytes, size_t v_token_bytes, size_t cap,
    bool is_swa, int32_t min_p, size_t valid_ctx, size_t hkv,
    size_t Dh, float* out_h) {

  constexpr size_t Bc = 32;
  size_t n_blocks = Dh / 32;

  float out_accum_buf[256];
  std::vector<float> dyn_accum;
  float* out_accum = out_accum_buf;
  if (Dh > 256) {
    dyn_accum.assign(Dh, 0.0f);
    out_accum = dyn_accum.data();
  } else {
    std::memset(out_accum_buf, 0, Dh * sizeof(float));
  }
  float m_prev = -1e9f;
  float l_prev = 0.0f;

  for (size_t c = 0; c < valid_ctx; c += Bc) {
    size_t k_tile = std::min(Bc, valid_ctx - c);
    float tile_scores[Bc];
    float m_tile = -1e9f;

    // 1. Q . K dot product for tokens in tile directly on q8_0 blocks
    for (size_t i = 0; i < k_tile; ++i) {
      int32_t p_key = min_p + int32_t(c + i);
      size_t slot = is_swa ? size_t(p_key % int32_t(cap)) : size_t(p_key);
      const block_q8_0* k_blocks = reinterpret_cast<const block_q8_0*>(
          k_cache + slot * k_token_bytes + hkv * n_blocks * sizeof(block_q8_0));

      float dot = 0.0f;
      for (size_t b = 0; b < n_blocks; ++b) {
        float d = half_to_float(k_blocks[b].d);
        const int8_t* qs = k_blocks[b].qs;
        const float* q_sub = q_h + b * 32;
        float bdot = 0.0f;
        for (int j = 0; j < 32; ++j) {
          bdot += q_sub[j] * float(qs[j]);
        }
        dot += bdot * d;
      }
      tile_scores[i] = dot;
      if (dot > m_tile) m_tile = dot;
    }

    // 2. Online softmax update
    float m_new = std::max(m_prev, m_tile);
    float alpha = std::exp(m_prev - m_new);
    float l_new = l_prev * alpha;

    for (size_t d = 0; d < Dh; ++d) {
      out_accum[d] *= alpha;
    }

    // 3. Tile accumulation P_tile . V_tile directly from q4_0
    for (size_t i = 0; i < k_tile; ++i) {
      float p_i = std::exp(tile_scores[i] - m_new);
      l_new += p_i;

      int32_t p_key = min_p + int32_t(c + i);
      size_t slot = is_swa ? size_t(p_key % int32_t(cap)) : size_t(p_key);
      const block_q4_0* v_blocks = reinterpret_cast<const block_q4_0*>(
          v_cache + slot * v_token_bytes + hkv * n_blocks * sizeof(block_q4_0));

      for (size_t b = 0; b < n_blocks; ++b) {
        float vd = half_to_float(v_blocks[b].d) * p_i;
        const uint8_t* qs = v_blocks[b].qs;
        float* out_sub = out_accum + b * 32;
        for (int j = 0; j < 16; ++j) {
          out_sub[j]      += vd * float(int(qs[j] & 0x0F) - 8);
          out_sub[16 + j] += vd * float(int(qs[j] >> 4) - 8);
        }
      }
    }

    m_prev = m_new;
    l_prev = l_new;
  }

  // 4. Final normalization
  float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
  for (size_t d = 0; d < Dh; ++d) {
    out_h[d] = out_accum[d] * inv_l;
  }
}

void Gemma4Engine::unload_model() {
  if (!model_loaded_) return;
  token_embd_ = nullptr;
  output_norm_ = nullptr;
  output_weight_ = nullptr;
  rope_freqs_ = nullptr;
  ple_token_embd_ = nullptr;
  ple_model_proj_ = nullptr;
  ple_proj_norm_ = nullptr;

  for (auto& lw : layers_) {
    lw.attn_norm = nullptr;
    lw.attn_q = nullptr;
    lw.attn_q_norm = nullptr;
    lw.attn_k = nullptr;
    lw.attn_k_norm = nullptr;
    lw.attn_v = nullptr;
    lw.attn_out = nullptr;
    lw.post_attn_norm = nullptr;
    lw.ffn_norm = nullptr;
    lw.ffn_gate = nullptr;
    lw.ffn_up = nullptr;
    lw.ffn_down = nullptr;
    lw.post_ffw_norm = nullptr;
    lw.inp_gate = nullptr;
    lw.ple_proj = nullptr;
    lw.post_ple_norm = nullptr;
    lw.layer_scale = nullptr;
  }

  if (compact_ple_.mapped_ptr && compact_ple_.mapped_ptr != MAP_FAILED) {
    munmap(compact_ple_.mapped_ptr, compact_ple_.mapped_size);
    compact_ple_.mapped_ptr = nullptr;
    compact_ple_.is_loaded = false;
  }

  owned_file_.reset();
  preloaded_weights_.clear();
  tensor_preloaded_ptr_.clear();
  act_arena_.clear();
  act_arena_.shrink_to_fit();
  scores_buf_.clear();
  scores_buf_.shrink_to_fit();
  model_loaded_ = false;
  logi("gemma4: unloaded model weights from RAM");
}

void Gemma4Engine::unload_kv() {
  if (!kv_loaded_) return;
  kv_arena_.clear();
  kv_arena_.shrink_to_fit();
  for (auto& s : slots_) {
    s.kv_arena.clear();
    s.kv_arena.shrink_to_fit();
    s.seq_pos = 0;
    s.snapshot.arena_bytes.clear();
    s.snapshot.arena_bytes.shrink_to_fit();
    s.snapshot.is_primed = false;
  }
  snapshot_.arena_bytes.clear();
  snapshot_.arena_bytes.shrink_to_fit();
  snapshot_.is_primed = false;
  kv_loaded_ = false;
  logi("gemma4: unloaded KV caches and snapshots from RAM");
}

void Gemma4Engine::ensure_model_loaded() {
  if (model_loaded_) return;
  if (model_path_.empty()) return;
  double t0 = now_ms();
  owned_file_ = std::make_unique<gguf::File>(model_path_);
  load_model(*owned_file_, max_context_, preload_amx_);
  if (!ple_path_.empty()) {
    load_ple_safetensors(ple_path_);
  }
  logi("gemma4: reloaded model in %.1f ms", now_ms() - t0);
}

void Gemma4Engine::ensure_kv_loaded(uint32_t) {
  if (kv_loaded_) return;
  ensure_model_loaded();
  size_t total_kv_bytes = 0;
  for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
    if (layers_[l].has_kv) {
      total_kv_bytes += layers_[l].kv_capacity * (layers_[l].k_token_bytes + layers_[l].v_token_bytes);
    }
  }
  kv_arena_.assign(total_kv_bytes, 0);
  for (auto& s : slots_) {
    s.kv_arena.assign(total_kv_bytes, 0);
    s.seq_pos = 0;
  }
  kv_loaded_ = true;

  if (!primer_tokens_.empty()) {
    prime_system_prompt(primer_tokens_, 0);
  }
  logi("gemma4: re-initialized KV cache arenas (%.1f MB)",
       (slots_.size() * total_kv_bytes) / (1024.0 * 1024.0));
}

Gemma4Engine::Gemma4Engine() = default;

Gemma4Engine::~Gemma4Engine() {
  if (compact_ple_.mapped_ptr && compact_ple_.mapped_ptr != MAP_FAILED) {
    munmap(compact_ple_.mapped_ptr, compact_ple_.mapped_size);
  }
}

bool Gemma4Engine::load_ple_safetensors(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    loge("gemma4: cannot open compact PLE safetensors %s", path.c_str());
    return false;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return false;
  }
  size_t sz = size_t(st.st_size);
  void* ptr = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    loge("gemma4: mmap failed for %s", path.c_str());
    return false;
  }
#if defined(MADV_WILLNEED)
  madvise(ptr, sz, MADV_WILLNEED);
#endif

  compact_ple_.mapped_ptr = ptr;
  compact_ple_.mapped_size = sz;

  uint64_t hsz = 0;
  std::memcpy(&hsz, ptr, 8);
  if (8 + hsz > sz) {
    loge("gemma4: invalid safetensors header length");
    return false;
  }

  std::string_view hstr(reinterpret_cast<const char*>(ptr) + 8, size_t(hsz));
  nlohmann::json j = nlohmann::json::parse(hstr);

  if (j.contains("__metadata__")) {
    const auto& meta = j["__metadata__"];
    if (meta.contains("embed_scale")) compact_ple_.embed_scale = std::stof(meta["embed_scale"].get<std::string>());
    if (meta.contains("num_layers")) compact_ple_.num_layers = uint32_t(std::stoi(meta["num_layers"].get<std::string>()));
    if (meta.contains("num_groups")) compact_ple_.num_groups = uint32_t(std::stoi(meta["num_groups"].get<std::string>()));
    if (meta.contains("group_size")) compact_ple_.group_size = uint32_t(std::stoi(meta["group_size"].get<std::string>()));
    if (meta.contains("num_codes")) compact_ple_.num_codes = uint32_t(std::stoi(meta["num_codes"].get<std::string>()));
    if (meta.contains("ple_dim")) compact_ple_.ple_dim = uint32_t(std::stoi(meta["ple_dim"].get<std::string>()));
    if (meta.contains("vocab_size")) compact_ple_.vocab_size = uint32_t(std::stoi(meta["vocab_size"].get<std::string>()));
  }

  const uint8_t* data_base = reinterpret_cast<const uint8_t*>(ptr) + 8 + hsz;

  if (!j.contains("codebooks") || !j.contains("codes_packed")) {
    loge("gemma4: missing codebooks / codes_packed in %s", path.c_str());
    return false;
  }

  size_t cb_off = j["codebooks"]["data_offsets"][0].get<size_t>();
  size_t cp_off = j["codes_packed"]["data_offsets"][0].get<size_t>();

  compact_ple_.codebooks = reinterpret_cast<const uint16_t*>(data_base + cb_off);
  compact_ple_.codes_packed = data_base + cp_off;
  compact_ple_.is_loaded = true;

  logi("gemma4: loaded compact AQLM PLE (%u layers, %u groups, %u codes, scale=%.1f, %.1f MB mapped)",
       compact_ple_.num_layers, compact_ple_.num_groups, compact_ple_.num_codes,
       compact_ple_.embed_scale, sz / (1024.0 * 1024.0));
  return true;
}

bool Gemma4Engine::load_model(const gguf::File& file, uint32_t max_context, bool preload_amx) {
  cfg_.max_context = max_context;
  cfg_.dim = uint32_t(file.num("gemma4.embedding_length", file.num("general.embedding_length", 1536)));
  cfg_.n_layers = uint32_t(file.num("gemma4.block_count", 35));
  cfg_.ffn_dim = uint32_t(file.num("gemma4.feed_forward_length", 6144));
  cfg_.n_heads = uint32_t(file.num("gemma4.attention.head_count", 8));
  cfg_.n_kv_heads = uint32_t(file.num("gemma4.attention.head_count_kv", 1));
  cfg_.head_dim = uint32_t(file.num("gemma4.attention.key_length", 256));
  cfg_.head_dim_swa = uint32_t(file.num("gemma4.attention.key_length_swa", cfg_.head_dim));
  cfg_.sliding_window = uint32_t(file.num("gemma4.attention.sliding_window", 512));
  cfg_.n_shared_kv = uint32_t(file.num("gemma4.attention.shared_kv_layers", 0));
  cfg_.ple_dim = uint32_t(file.num("gemma4.embedding_length_per_layer_input", 256));
  cfg_.rope_freq_base = float(file.real("gemma4.rope.freq_base", 1000000.0));
  cfg_.rope_freq_base_swa = float(file.real("gemma4.rope.freq_base_swa", 10000.0));
  cfg_.rms_norm_eps = float(file.real("gemma4.attention.layer_norm_rms_epsilon", 1e-6));
  cfg_.logit_softcap = float(file.real("gemma4.final_logit_softcapping", 30.0));

  token_embd_ = file.find("token_embd.weight");
  output_norm_ = file.find("output_norm.weight");
  output_weight_ = file.find("output.weight");
  if (!output_weight_) output_weight_ = token_embd_;

  rope_freqs_ = file.find("rope_freqs.weight");

  ple_token_embd_ = file.find("per_layer_token_embd.weight");
  ple_model_proj_ = file.find("per_layer_model_proj.weight");
  ple_proj_norm_ = file.find("per_layer_proj_norm.weight");

  if (!token_embd_ || !output_norm_) {
    loge("gemma4: missing essential global tensors (token_embd / output_norm)");
    return false;
  }

  cfg_.vocab_size = uint32_t(token_embd_->rows());
  last_logits_.assign(cfg_.vocab_size, 0.0f);

  layers_.resize(cfg_.n_layers);
  int last_non_shared_swa = -1;
  int last_non_shared_full = -1;

  bool has_swa_pattern = file.has("gemma4.attention.sliding_window_pattern");
  const auto& swa_pattern = has_swa_pattern ? file.meta()["gemma4.attention.sliding_window_pattern"] : nlohmann::json::array();

  for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
    LayerWeights& lw = layers_[l];
    std::string prefix = "blk." + std::to_string(l) + ".";

    lw.attn_norm      = file.find(prefix + "attn_norm.weight");
    lw.attn_q         = file.find(prefix + "attn_q.weight");
    lw.attn_q_norm    = file.find(prefix + "attn_q_norm.weight");
    lw.attn_k         = file.find(prefix + "attn_k.weight");
    lw.attn_k_norm    = file.find(prefix + "attn_k_norm.weight");
    lw.attn_v         = file.find(prefix + "attn_v.weight");
    lw.attn_out       = file.find(prefix + "attn_output.weight");
    lw.post_attn_norm = file.find(prefix + "post_attention_norm.weight");

    lw.ffn_norm       = file.find(prefix + "ffn_norm.weight");
    lw.ffn_gate       = file.find(prefix + "ffn_gate.weight");
    lw.ffn_up         = file.find(prefix + "ffn_up.weight");
    lw.ffn_down       = file.find(prefix + "ffn_down.weight");
    lw.post_ffw_norm  = file.find(prefix + "post_ffw_norm.weight");

    lw.inp_gate       = file.find(prefix + "inp_gate.weight");
    lw.ple_proj       = file.find(prefix + "proj.weight");
    lw.post_ple_norm  = file.find(prefix + "post_norm.weight");
    lw.layer_scale    = file.find(prefix + "layer_output_scale.weight");

    // SWA pattern: typically (l + 1) % 5 == 0 is full attention, else SWA
    if (has_swa_pattern && l < swa_pattern.size()) {
      lw.is_swa = swa_pattern[l].get<bool>();
    } else {
      lw.is_swa = ((l + 1) % 5 != 0);
    }

    lw.head_dim = lw.is_swa ? cfg_.head_dim_swa : cfg_.head_dim;
    if (lw.attn_q) {
      lw.q_dim = uint32_t(lw.attn_q->rows());
      lw.n_heads = lw.q_dim / lw.head_dim;
    }
    if (lw.attn_k) {
      lw.kv_dim = uint32_t(lw.attn_k->rows());
      lw.n_kv_heads = lw.kv_dim / lw.head_dim;
    }
    if (lw.ffn_gate) {
      lw.ffn_dim = uint32_t(lw.ffn_gate->rows());
    }

    // Shared KV: trailing n_shared_kv layers do not have their own KV projections
    bool is_shared = (cfg_.n_shared_kv > 0 && l >= cfg_.n_layers - cfg_.n_shared_kv) || (lw.attn_k == nullptr);
    lw.has_kv = !is_shared;

    if (lw.has_kv) {
      if (lw.is_swa) last_non_shared_swa = int(l);
      else last_non_shared_full = int(l);
    } else {
      lw.kv_src_layer = lw.is_swa ? last_non_shared_swa : last_non_shared_full;
      if (lw.kv_src_layer >= 0) {
        lw.kv_dim = layers_[lw.kv_src_layer].kv_dim;
        lw.n_kv_heads = layers_[lw.kv_src_layer].n_kv_heads;
      }
    }
  }

  // Setup contiguous KV cache arena (K: q8_0, V: q4_0)
  size_t total_kv_bytes = 0;
  for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
    LayerWeights& lw = layers_[l];
    if (lw.has_kv) {
      size_t cap = lw.is_swa ? cfg_.sliding_window : cfg_.max_context;
      lw.kv_capacity = cap;
      size_t n_blocks = lw.head_dim / 32;
      lw.k_token_bytes = lw.n_kv_heads * n_blocks * sizeof(block_q8_0);
      lw.v_token_bytes = lw.n_kv_heads * n_blocks * sizeof(block_q4_0);

      lw.k_offset = total_kv_bytes;
      total_kv_bytes += cap * lw.k_token_bytes;
      lw.v_offset = total_kv_bytes;
      total_kv_bytes += cap * lw.v_token_bytes;
    } else {
      int src = lw.kv_src_layer >= 0 ? lw.kv_src_layer : 0;
      lw.k_offset = layers_[src].k_offset;
      lw.v_offset = layers_[src].v_offset;
      lw.k_token_bytes = layers_[src].k_token_bytes;
      lw.v_token_bytes = layers_[src].v_token_bytes;
      lw.kv_capacity = layers_[src].kv_capacity;
    }
  }

  kv_arena_.assign(total_kv_bytes, 0);
  seq_pos_ = 0;

  // Pre-allocate static activation arena
  uint32_t max_q_dim = 2048;
  uint32_t max_kv_dim = 256;
  uint32_t max_ffn_dim = 6144;
  for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
    max_q_dim = std::max(max_q_dim, layers_[l].q_dim);
    max_kv_dim = std::max(max_kv_dim, layers_[l].kv_dim);
    max_ffn_dim = std::max(max_ffn_dim, layers_[l].ffn_dim);
  }

  size_t S = cfg_.prefill_step;
  size_t act_size = S * cfg_.dim * 6 +                   // x0, cur, norm_buf, attn_out, ffn_down, ple_out
                    S * max_q_dim * 2 +                  // Q, temp
                    S * max_kv_dim * 2 +                 // K, V
                    S * max_ffn_dim * 2 +                // ffn_gate, ffn_up
                    S * cfg_.n_layers * cfg_.ple_dim * 3 + // ple_buf, ple_proj, ple_combined
                    S * cfg_.ple_dim * 2;                // ple_gate, temp
  act_arena_.assign(act_size, 0.0f);

  // Auto-enable YaRN context extension if max_context > orig_ctx (e.g. 256K vs 128K)
  if (cfg_.max_context > cfg_.yarn.orig_ctx || cfg_.yarn.enabled) {
    cfg_.yarn.enabled = true;
    cfg_.yarn.target_ctx = cfg_.max_context;
    if (cfg_.yarn.scale <= 1.0f) {
      cfg_.yarn.scale = float(cfg_.max_context) / float(cfg_.yarn.orig_ctx);
    }
  }
  cfg_.yarn.update(cfg_.head_dim, cfg_.rope_freq_base);

  if (cfg_.yarn.enabled) {
    logi("gemma4: YaRN context extension active (orig_ctx=%u, target_ctx=%u, scale=%.2f, beta_fast=%.1f, beta_slow=%.1f, eff_mscale=%.4f, corr_dims=[%.0f, %.0f])",
         cfg_.yarn.orig_ctx, cfg_.yarn.target_ctx, cfg_.yarn.scale, cfg_.yarn.beta_fast, cfg_.yarn.beta_slow,
         cfg_.yarn.eff_mscale, cfg_.yarn.corr_dim_low, cfg_.yarn.corr_dim_high);
  }

  // Pre-allocate attention scores buffer up to max_context to prevent allocation churn
  scores_buf_.assign(cfg_.max_context, 0.0f);

  // Initialize >= 4 user slots with independent contiguous KV arenas
  init_slots(4);

  model_loaded_ = true;
  kv_loaded_ = true;
  max_context_ = max_context;
  preload_amx_ = preload_amx;

  logi("gemma4: loaded model (H=%u, L=%u, I_max=%u, Hq_max=%u, Hkv_max=%u, SWA=%u, shared_kv=%u, kv_arena=%.1f MB, act_arena=%.1f MB, k=%s, v=%s, flash_attn=%s)",
       cfg_.dim, cfg_.n_layers, max_ffn_dim, max_q_dim / cfg_.head_dim_swa, max_kv_dim / cfg_.head_dim_swa, cfg_.sliding_window,
       cfg_.n_shared_kv, (total_kv_bytes) / (1024.0 * 1024.0),
       (act_size * sizeof(float)) / (1024.0 * 1024.0),
       cfg_.kv_type_k.c_str(), cfg_.kv_type_v.c_str(), cfg_.flash_attn ? "yes" : "no");

  return true;
}

bool Gemma4Engine::init_slots(uint32_t num_slots) {
  if (num_slots < 1) num_slots = 4;
  slots_.resize(num_slots);
  for (uint32_t i = 0; i < num_slots; ++i) {
    UserSlot& s = slots_[i];
    s.slot_id = i;
    s.bound_user.clear();
    s.last_accessed_ms = 0;
    s.kv_arena = kv_arena_; // copy layout and zeroed elements
    s.seq_pos = 0;
    s.snapshot = snapshot_;
    s.attached_adapter_id.clear();
    s.is_active = false;
  }
  logi("gemma4: initialized %u user slots (%.1f MB KV cache total, k=%s, v=%s)",
       num_slots, (num_slots * kv_arena_.size()) / (1024.0 * 1024.0),
       cfg_.kv_type_k.c_str(), cfg_.kv_type_v.c_str());
  return true;
}

void Gemma4Engine::reset_state(uint32_t slot_idx) {
  if (slot_idx < slots_.size()) {
    std::fill(slots_[slot_idx].kv_arena.begin(), slots_[slot_idx].kv_arena.end(), uint8_t(0));
    slots_[slot_idx].seq_pos = 0;
  } else {
    std::fill(kv_arena_.begin(), kv_arena_.end(), uint8_t(0));
    seq_pos_ = 0;
  }
}

bool Gemma4Engine::prime_system_prompt(const std::vector<int32_t>& system_tokens, uint32_t slot_idx) {
  if (system_tokens.empty()) return false;
  double t0 = now_ms();
  reset_state(slot_idx);
  prefill(slot_idx, system_tokens, /*keep_logits=*/false);

  primer_tokens_ = system_tokens;
  KVSnapshot snap;
  if (slot_idx < slots_.size()) {
    snap.arena_bytes = slots_[slot_idx].kv_arena;
    snap.primer_ids = system_tokens;
    snap.seq_pos = slots_[slot_idx].seq_pos;
    snap.is_primed = true;
    slots_[slot_idx].snapshot = snap;
  }
  snapshot_ = snap;
  if (slot_idx == 0) {
    for (auto& s : slots_) {
      if (!s.snapshot.is_primed) {
        s.snapshot = snap;
      }
    }
  }

  logi("gemma4: system prompt primed on slot %u (%zu tokens in %.1f ms, snapshot %.1f MB)",
       slot_idx, system_tokens.size(), now_ms() - t0,
       (snap.arena_bytes.size()) / (1024.0 * 1024.0));
  return true;
}

bool Gemma4Engine::restore_snapshot(uint32_t slot_idx) {
  if (slot_idx < slots_.size()) {
    UserSlot& s = slots_[slot_idx];
    const KVSnapshot& snap = s.snapshot.is_primed ? s.snapshot : snapshot_;
    if (!snap.is_primed) return false;
    std::memcpy(s.kv_arena.data(), snap.arena_bytes.data(), s.kv_arena.size());
    s.seq_pos = snap.seq_pos;
    return true;
  }
  if (!snapshot_.is_primed) return false;
  std::memcpy(kv_arena_.data(), snapshot_.arena_bytes.data(), kv_arena_.size());
  seq_pos_ = snapshot_.seq_pos;
  return true;
}

bool Gemma4Engine::register_lora_adapter(std::unique_ptr<LoraAdapter> adapter) {
  if (!adapter) return false;
  logi("gemma4: registered LoRA adapter '%s' (%zu layers)",
       adapter->id.c_str(), adapter->layers.size());
  for (auto& existing : adapters_) {
    if (existing && existing->id == adapter->id) {
      existing = std::move(adapter);
      return true;
    }
  }
  adapters_.push_back(std::move(adapter));
  return true;
}

bool Gemma4Engine::attach_adapter_to_slot(uint32_t slot_idx, const std::string& adapter_id) {
  if (slot_idx >= slots_.size()) return false;
  if (!adapter_id.empty() && !get_adapter(adapter_id)) {
    logi("gemma4: attaching adapter ID '%s' to slot %u (pending load)", adapter_id.c_str(), slot_idx);
  }
  slots_[slot_idx].attached_adapter_id = adapter_id;
  logi("gemma4: slot %u attached adapter '%s'",
       slot_idx, adapter_id.empty() ? "(none/base)" : adapter_id.c_str());
  return true;
}

const LoraAdapter* Gemma4Engine::get_adapter(const std::string& id) const {
  if (id.empty()) return nullptr;
  for (const auto& a : adapters_) {
    if (a && a->id == id) return a.get();
  }
  return nullptr;
}

std::vector<std::string> Gemma4Engine::list_adapters() const {
  std::vector<std::string> res;
  for (const auto& a : adapters_) {
    if (a) res.push_back(a->id);
  }
  return res;
}

bool Gemma4Engine::load_lora_adapter(const std::string& id, const std::string& path, float default_alpha) {
  try {
    std::ifstream test_f(path, std::ios::binary);
    if (!test_f.good()) {
      loge("gemma4: cannot open adapter file %s", path.c_str());
      return false;
    }
    uint32_t magic = 0;
    test_f.read(reinterpret_cast<char*>(&magic), 4);
    test_f.close();

    auto adapter = std::make_unique<LoraAdapter>();
    adapter->id = id;
    adapter->default_alpha = default_alpha;
    adapter->layers.resize(cfg_.n_layers);

    if (magic == 0x46554747) { // "GGUF" in little endian
      gguf::File gf(path);
      float alpha = default_alpha;
      if (gf.has("adapter.lora.alpha")) {
        alpha = float(gf.real("adapter.lora.alpha", default_alpha));
      }

      for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        std::string pfx = "blk." + std::to_string(l) + ".";
        auto load_pair = [&](const std::string& mod_name, std::unique_ptr<LoraTensor>& target) {
          const gguf::Tensor* ta = gf.find(pfx + mod_name + ".weight.lora_a");
          if (!ta) ta = gf.find(pfx + mod_name + ".lora_a");
          const gguf::Tensor* tb = gf.find(pfx + mod_name + ".weight.lora_b");
          if (!tb) tb = gf.find(pfx + mod_name + ".lora_b");

          if (ta && tb) {
            target = std::make_unique<LoraTensor>();
            target->alpha = alpha;
            target->d_in = int(ta->ne[0]);
            target->r = int(ta->ne[1]);
            target->d_out = int(tb->ne[1]);
            target->scale = target->r > 0 ? (alpha / float(target->r)) : 1.0f;
            target->a.resize(target->r * target->d_in);
            target->b.resize(target->d_out * target->r);

            dequantize(ta->type, ta->data, target->a.data(), target->a.size());
            dequantize(tb->type, tb->data, target->b.data(), target->b.size());
          }
        };

        load_pair("attn_q", adapter->layers[l].attn_q);
        load_pair("attn_k", adapter->layers[l].attn_k);
        load_pair("attn_v", adapter->layers[l].attn_v);
        load_pair("attn_output", adapter->layers[l].attn_out);
        load_pair("ffn_gate", adapter->layers[l].ffn_gate);
        load_pair("ffn_up", adapter->layers[l].ffn_up);
        load_pair("ffn_down", adapter->layers[l].ffn_down);
      }
      return register_lora_adapter(std::move(adapter));
    }

    // Check if Safetensors
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
      struct stat st;
      fstat(fd, &st);
      size_t fsz = size_t(st.st_size);
      void* mptr = mmap(nullptr, fsz, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);
      if (mptr != MAP_FAILED) {
        uint64_t hlen = 0;
        std::memcpy(&hlen, mptr, 8);
        if (8 + hlen <= fsz) {
          std::string_view json_sv(reinterpret_cast<const char*>(mptr) + 8, hlen);
          nlohmann::json sj = nlohmann::json::parse(json_sv);
          const uint8_t* data_base = reinterpret_cast<const uint8_t*>(mptr) + 8 + hlen;
          float alpha = default_alpha;

          auto extract_tensor = [&](const std::string& name, std::vector<float>& out, int& d0, int& d1) -> bool {
            if (!sj.contains(name)) return false;
            const auto& tinfo = sj[name];
            std::string dtype = tinfo.value("dtype", "F32");
            const auto& shape = tinfo["shape"];
            if (shape.size() < 2) return false;
            d0 = int(shape[0]);
            d1 = int(shape[1]);
            size_t n_elem = size_t(d0 * d1);
            out.resize(n_elem);
            size_t off_start = size_t(tinfo["data_offsets"][0]);
            const uint8_t* raw = data_base + off_start;

            if (dtype == "F32") {
              std::memcpy(out.data(), raw, n_elem * sizeof(float));
            } else if (dtype == "F16") {
              dequantize(0x01, raw, out.data(), n_elem); // F16
            } else if (dtype == "BF16") {
              dequantize(0x1e, raw, out.data(), n_elem); // BF16
            } else {
              return false;
            }
            return true;
          };

          for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
            std::string l_str = std::to_string(l);
            auto load_hf_pair = [&](const std::string& mod_key, std::unique_ptr<LoraTensor>& target) {
              std::string kA = "base_model.model.model.layers." + l_str + "." + mod_key + ".lora_A.weight";
              std::string kB = "base_model.model.model.layers." + l_str + "." + mod_key + ".lora_B.weight";
              if (!sj.contains(kA)) {
                kA = "model.layers." + l_str + "." + mod_key + ".lora_a.weight";
                kB = "model.layers." + l_str + "." + mod_key + ".lora_b.weight";
              }
              std::vector<float> a_data, b_data;
              int a_r = 0, a_din = 0, b_dout = 0, b_r = 0;
              if (extract_tensor(kA, a_data, a_r, a_din) && extract_tensor(kB, b_data, b_dout, b_r)) {
                target = std::make_unique<LoraTensor>();
                target->r = a_r;
                target->d_in = a_din;
                target->d_out = b_dout;
                target->alpha = alpha;
                target->scale = target->r > 0 ? (alpha / float(target->r)) : 1.0f;
                target->a = std::move(a_data);
                target->b = std::move(b_data);
              }
            };

            load_hf_pair("self_attn.q_proj", adapter->layers[l].attn_q);
            load_hf_pair("self_attn.k_proj", adapter->layers[l].attn_k);
            load_hf_pair("self_attn.v_proj", adapter->layers[l].attn_v);
            load_hf_pair("self_attn.o_proj", adapter->layers[l].attn_out);
            load_hf_pair("mlp.gate_proj", adapter->layers[l].ffn_gate);
            load_hf_pair("mlp.up_proj", adapter->layers[l].ffn_up);
            load_hf_pair("mlp.down_proj", adapter->layers[l].ffn_down);
          }
          munmap(mptr, fsz);
          return register_lora_adapter(std::move(adapter));
        }
        munmap(mptr, fsz);
      }
    }
  } catch (const std::exception& e) {
    loge("gemma4: failed to load adapter from %s: %s", path.c_str(), e.what());
  }
  return false;
}

void Gemma4Engine::apply_lora(const float* x, const LoraTensor& lora, float* y, size_t T) {
  if (lora.r <= 0 || lora.a.empty() || lora.b.empty()) return;
  int r = lora.r;
  int d_in = lora.d_in;
  int d_out = lora.d_out;
  float scale = lora.scale;

  if (T == 1) {
    // Single token decode: vector-matrix dot products
    std::vector<float> h(r, 0.0f);
    for (int i = 0; i < r; ++i) {
      vDSP_dotpr(x, 1, lora.a.data() + i * d_in, 1, &h[i], d_in);
    }
    for (int j = 0; j < d_out; ++j) {
      float delta = 0.0f;
      vDSP_dotpr(lora.b.data() + j * r, 1, h.data(), 1, &delta, r);
      y[j] += scale * delta;
    }
  } else {
    // Multi-token prefill: Apple Accelerate AMX cblas_sgemm
    std::vector<float> H(T * r, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                int(T), r, d_in,
                1.0f, x, d_in,
                lora.a.data(), d_in,
                0.0f, H.data(), r);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                int(T), d_out, r,
                scale, H.data(), r,
                lora.b.data(), r,
                1.0f, y, d_out);
  }
}

void Gemma4Engine::matmul(const float* x, const gguf::Tensor* w, float* y, size_t M, size_t N, size_t K) {
  if (!w || !w->data) {
    std::memset(y, 0, M * N * sizeof(float));
    return;
  }

  // Fast path for 1xK vector-matrix dot products (Decode step)
  if (M == 1) {
    uint32_t type = w->type;
    const uint8_t* base = w->data;
    uint64_t row_bytes = w->row_bytes();
    float* y_dst = y;

    dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
      const uint8_t* row_src = base + j * row_bytes;
      std::vector<float> row_dequant(K);
      dequantize(type, row_src, row_dequant.data(), K);
      float dot = 0.0f;
      vDSP_dotpr(x, 1, row_dequant.data(), 1, &dot, K);
      y_dst[j] = dot;
    });
    return;
  }

  // Prefill path (M > 1): dequantize weight rows into reusable buffer and invoke Apple AMX cblas_sgemm
  uint32_t type = w->type;
  std::vector<float> w_dequant(N * K);
  const uint8_t* base = w->data;
  uint64_t row_bytes = w->row_bytes();
  float* w_dst = w_dequant.data();

  dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
    dequantize(type, base + j * row_bytes, w_dst + j * K, K);
  });

  // C = A * B^T -> y (M x N) = x (M x K) * w^T (K x N)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              int(M), int(N), int(K),
              1.0f, x, int(K),
              w_dequant.data(), int(K),
              0.0f, y, int(N));
}

void Gemma4Engine::lookup_token_embd(const int32_t* tokens, size_t n_tokens, float* out) {
  if (!token_embd_ || !token_embd_->data) return;
  uint32_t type = token_embd_->type;
  const uint8_t* base = token_embd_->data;
  uint64_t row_bytes = token_embd_->row_bytes();
  size_t dim = cfg_.dim;

  for (size_t t = 0; t < n_tokens; ++t) {
    int32_t id = tokens[t];
    if (id >= 0 && id < int32_t(cfg_.vocab_size)) {
      dequantize(type, base + id * row_bytes, out + t * dim, dim);
    } else {
      std::memset(out + t * dim, 0, dim * sizeof(float));
    }
  }
}

void Gemma4Engine::lookup_ple_embd(const int32_t* tokens, size_t n_tokens, float* out) {
  if (compact_ple_.is_loaded) {
    size_t total_ple_dim = cfg_.n_layers * cfg_.ple_dim;
    const uint8_t* cp = compact_ple_.codes_packed;
    const uint16_t* cb = compact_ple_.codebooks;
    float scale = compact_ple_.embed_scale;

    for (size_t t = 0; t < n_tokens; ++t) {
      int32_t id = tokens[t];
      if (id < 0 || id >= int32_t(compact_ple_.vocab_size)) {
        std::memset(out + t * total_ple_dim, 0, total_ple_dim * sizeof(float));
        continue;
      }
      float* out_tok = out + t * total_ple_dim;
      for (size_t l = 0; l < compact_ple_.num_layers; ++l) {
        for (size_t g = 0; g < compact_ple_.num_groups; ++g) {
          size_t idx = (l * 32 + g) * 262144 + size_t(id);
          size_t start_bit = idx * 7;
          size_t b = start_bit / 8;
          size_t o = start_bit % 8;
          uint16_t w = uint16_t(cp[b]) | (uint16_t(cp[b + 1]) << 8);
          uint8_t code = (w >> o) & 0x7F;

          const uint16_t* cb_entry = cb + (l * 32 * 128 + g * 128 + code) * 8;
          for (size_t d = 0; d < 8; ++d) {
            out_tok[l * 256 + g * 8 + d] = bf16_to_f32(cb_entry[d]) * scale;
          }
        }
      }
    }
    return;
  }
  if (!ple_token_embd_ || !ple_token_embd_->data) {
    std::memset(out, 0, n_tokens * cfg_.n_layers * cfg_.ple_dim * sizeof(float));
    return;
  }
  uint32_t type = ple_token_embd_->type;
  const uint8_t* base = ple_token_embd_->data;
  uint64_t row_bytes = ple_token_embd_->row_bytes();
  size_t total_ple_dim = cfg_.n_layers * cfg_.ple_dim;

  for (size_t t = 0; t < n_tokens; ++t) {
    int32_t id = tokens[t];
    if (id >= 0 && id < int32_t(cfg_.vocab_size)) {
      dequantize(type, base + id * row_bytes, out + t * total_ple_dim, total_ple_dim);
    } else {
      std::memset(out + t * total_ple_dim, 0, total_ple_dim * sizeof(float));
    }
  }
}

void Gemma4Engine::forward_chunk(uint32_t slot_idx, const int32_t* tokens, size_t T, int32_t start_pos, bool compute_logits) {
  double t_chunk_start = now_ms();
  size_t H = cfg_.dim;
  size_t L = cfg_.n_layers;
  size_t Dple = cfg_.ple_dim;

  std::vector<uint8_t>& active_kv = (slot_idx < slots_.size()) ? slots_[slot_idx].kv_arena : kv_arena_;
  const LoraAdapter* adapter = nullptr;
  if (slot_idx < slots_.size() && !slots_[slot_idx].attached_adapter_id.empty()) {
    adapter = get_adapter(slots_[slot_idx].attached_adapter_id);
  }

  uint32_t max_q_dim = 2048;
  uint32_t max_kv_dim = 256;
  uint32_t max_ffn_dim = 6144;
  for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
    max_q_dim = std::max(max_q_dim, layers_[l].q_dim);
    max_kv_dim = std::max(max_kv_dim, layers_[l].kv_dim);
    max_ffn_dim = std::max(max_ffn_dim, layers_[l].ffn_dim);
  }

  float* x0 = act_arena_.data();
  float* cur = x0 + T * H;
  float* norm_buf = cur + T * H;
  float* Q = norm_buf + T * H;
  float* K_cur = Q + T * max_q_dim;
  float* V_cur = K_cur + T * max_kv_dim;
  float* attn_out = V_cur + T * max_kv_dim;
  float* ffn_gate = attn_out + T * max_q_dim;
  float* ffn_up = ffn_gate + T * max_ffn_dim;
  float* ffn_down = ffn_up + T * max_ffn_dim;
  float* ple_buf = ffn_down + T * H;
  float* ple_proj = ple_buf + T * L * Dple;
  float* ple_combined = ple_proj + T * L * Dple;
  float* ple_gate = ple_combined + T * L * Dple;
  float* ple_out = ple_gate + T * Dple;

  // 1. Token embeddings
  lookup_token_embd(tokens, T, x0);
  float sqrt_H = std::sqrt(float(H));
  vDSP_vsmul(x0, 1, &sqrt_H, cur, 1, T * H);

  // 2. Per-layer embeddings (PLE)
  if (ple_token_embd_ || compact_ple_.is_loaded) {
    lookup_ple_embd(tokens, T, ple_buf);
    if (!compact_ple_.is_loaded) {
      float sqrt_ple = std::sqrt(float(Dple));
      vDSP_vsmul(ple_buf, 1, &sqrt_ple, ple_buf, 1, T * L * Dple);
    }

    // ple_proj = per_layer_model_proj @ (inputs_embeds / sqrt(H)) = per_layer_model_proj @ x0
    matmul(x0, ple_model_proj_, ple_proj, T, L * Dple, H);

    // per-layer projection norm + combination
    std::vector<float> proj_norm_w(Dple, 1.0f);
    if (ple_proj_norm_) {
      dequantize(ple_proj_norm_->type, ple_proj_norm_->data, proj_norm_w.data(), Dple);
    }

    float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
    for (size_t t = 0; t < T; ++t) {
      for (size_t l = 0; l < L; ++l) {
        float* p_proj = ple_proj + (t * L + l) * Dple;
        float* p_tok  = ple_buf  + (t * L + l) * Dple;
        float* p_out  = ple_combined + (t * L + l) * Dple;
        rms_norm(p_proj, proj_norm_w.data(), p_proj, Dple, cfg_.rms_norm_eps);
        for (size_t d = 0; d < Dple; ++d) {
          p_out[d] = (p_proj[d] + p_tok[d]) * inv_sqrt_2;
        }
      }
    }
  }

  // 3. Transformer layers
  double t_layers_start = now_ms();
  for (uint32_t l = 0; l < L; ++l) {
    double t_layer_start = now_ms();
    LayerWeights& lw = layers_[l];
    size_t Hq = lw.n_heads;
    size_t Hkv = lw.n_kv_heads;
    size_t Dh = lw.head_dim;
    size_t I = lw.ffn_dim;
    size_t q_dim = lw.q_dim;
    size_t kv_dim = lw.kv_dim;

    std::vector<float> attn_norm_w(H, 1.0f);
    if (lw.attn_norm) dequantize(lw.attn_norm->type, lw.attn_norm->data, attn_norm_w.data(), H);

    for (size_t t = 0; t < T; ++t) {
      rms_norm(cur + t * H, attn_norm_w.data(), norm_buf + t * H, H, cfg_.rms_norm_eps);
    }

    // Q projection (Base + LoRA)
    matmul(norm_buf, lw.attn_q, Q, T, q_dim, H);
    if (adapter && l < adapter->layers.size() && adapter->layers[l].attn_q) {
      apply_lora(norm_buf, *adapter->layers[l].attn_q, Q, T);
    }

    std::vector<float> q_norm_w(Dh, 1.0f);
    if (lw.attn_q_norm) dequantize(lw.attn_q_norm->type, lw.attn_q_norm->data, q_norm_w.data(), Dh);

    const float* rf_data = (!lw.is_swa && rope_freqs_) ? reinterpret_cast<const float*>(rope_freqs_->data) : nullptr;
    for (size_t t = 0; t < T; ++t) {
      int32_t pos = start_pos + int32_t(t);
      float fbase = lw.is_swa ? cfg_.rope_freq_base_swa : cfg_.rope_freq_base;
      bool use_yarn = (!lw.is_swa && cfg_.yarn.enabled);
      for (size_t h = 0; h < Hq; ++h) {
        float* q_head = Q + (t * Hq + h) * Dh;
        rms_norm(q_head, q_norm_w.data(), q_head, Dh, cfg_.rms_norm_eps);
        apply_rope_yarn(q_head, Dh, pos, fbase, cfg_.yarn, use_yarn, rf_data);
      }
    }

    // K & V projections (if this layer owns its KV cache)
    if (lw.has_kv) {
      matmul(norm_buf, lw.attn_k, K_cur, T, kv_dim, H);
      if (adapter && l < adapter->layers.size() && adapter->layers[l].attn_k) {
        apply_lora(norm_buf, *adapter->layers[l].attn_k, K_cur, T);
      }

      if (lw.attn_v) {
        matmul(norm_buf, lw.attn_v, V_cur, T, kv_dim, H);
        if (adapter && l < adapter->layers.size() && adapter->layers[l].attn_v) {
          apply_lora(norm_buf, *adapter->layers[l].attn_v, V_cur, T);
        }
      } else {
        std::memcpy(V_cur, K_cur, T * kv_dim * sizeof(float));
      }

      std::vector<float> k_norm_w(Dh, 1.0f);
      if (lw.attn_k_norm) dequantize(lw.attn_k_norm->type, lw.attn_k_norm->data, k_norm_w.data(), Dh);

      const float* rf_data = (!lw.is_swa && rope_freqs_) ? reinterpret_cast<const float*>(rope_freqs_->data) : nullptr;
      for (size_t t = 0; t < T; ++t) {
        int32_t pos = start_pos + int32_t(t);
        float fbase = lw.is_swa ? cfg_.rope_freq_base_swa : cfg_.rope_freq_base;
        bool use_yarn = (!lw.is_swa && cfg_.yarn.enabled);
        for (size_t h = 0; h < Hkv; ++h) {
          float* k_head = K_cur + (t * Hkv + h) * Dh;
          float* v_head = V_cur + (t * Hkv + h) * Dh;
          rms_norm(k_head, k_norm_w.data(), k_head, Dh, cfg_.rms_norm_eps);
          apply_rope_yarn(k_head, Dh, pos, fbase, cfg_.yarn, use_yarn, rf_data);
          rms_norm(v_head, nullptr, v_head, Dh, cfg_.rms_norm_eps); // unweighted V norm
        }

        // Write to contiguous active_kv (K: q8_0, V: q4_0)
        size_t slot = lw.is_swa ? size_t(pos % int32_t(lw.kv_capacity)) : size_t(pos);
        if (slot < lw.kv_capacity) {
          uint8_t* k_dst = active_kv.data() + lw.k_offset + slot * lw.k_token_bytes;
          uint8_t* v_dst = active_kv.data() + lw.v_offset + slot * lw.v_token_bytes;
          quantize_q8_0(K_cur + t * kv_dim, reinterpret_cast<block_q8_0*>(k_dst), kv_dim);
          quantize_q4_0(V_cur + t * kv_dim, reinterpret_cast<block_q4_0*>(v_dst), kv_dim);
        }
      }
    }

    // Attention calculation using Flash Attention on CPU (tiled online softmax)
    // Avoids materializing the attention score matrix in RAM (critical at 256K context)
    const uint8_t* k_cache = active_kv.data() + lw.k_offset;
    const uint8_t* v_cache = active_kv.data() + lw.v_offset;
    size_t cap = lw.kv_capacity;

    std::memset(attn_out, 0, T * q_dim * sizeof(float));

    for (size_t t = 0; t < T; ++t) {
      int32_t pos = start_pos + int32_t(t);
      int32_t min_p = lw.is_swa ? std::max(0, pos - int32_t(cfg_.sliding_window) + 1) : 0;
      size_t valid_ctx = size_t(pos - min_p + 1);

      for (size_t hq = 0; hq < Hq; ++hq) {
        size_t hkv = hq / (Hq / Hkv);
        const float* q_h = Q + (t * Hq + hq) * Dh;
        float* out_h = attn_out + (t * Hq + hq) * Dh;

        flash_attn_head(q_h, k_cache, v_cache, lw.k_token_bytes, lw.v_token_bytes,
                        cap, lw.is_swa, min_p, valid_ctx, hkv, Dh, out_h);
      }
    }

    // Post-attention norm and output projection (Base + LoRA)
    std::vector<float> post_attn_w(H, 1.0f);
    if (lw.post_attn_norm) dequantize(lw.post_attn_norm->type, lw.post_attn_norm->data, post_attn_w.data(), H);

    matmul(attn_out, lw.attn_out, norm_buf, T, H, q_dim);
    if (adapter && l < adapter->layers.size() && adapter->layers[l].attn_out) {
      apply_lora(attn_out, *adapter->layers[l].attn_out, norm_buf, T);
    }

    for (size_t t = 0; t < T; ++t) {
      rms_norm(norm_buf + t * H, post_attn_w.data(), norm_buf + t * H, H, cfg_.rms_norm_eps);
      vDSP_vadd(cur + t * H, 1, norm_buf + t * H, 1, cur + t * H, 1, H);
    }

    // FFN: GeGLU (Base + LoRA)
    std::vector<float> ffn_norm_w(H, 1.0f);
    if (lw.ffn_norm) dequantize(lw.ffn_norm->type, lw.ffn_norm->data, ffn_norm_w.data(), H);

    for (size_t t = 0; t < T; ++t) {
      rms_norm(cur + t * H, ffn_norm_w.data(), norm_buf + t * H, H, cfg_.rms_norm_eps);
    }

    matmul(norm_buf, lw.ffn_gate, ffn_gate, T, I, H);
    if (adapter && l < adapter->layers.size() && adapter->layers[l].ffn_gate) {
      apply_lora(norm_buf, *adapter->layers[l].ffn_gate, ffn_gate, T);
    }

    matmul(norm_buf, lw.ffn_up, ffn_up, T, I, H);
    if (adapter && l < adapter->layers.size() && adapter->layers[l].ffn_up) {
      apply_lora(norm_buf, *adapter->layers[l].ffn_up, ffn_up, T);
    }

    for (size_t t = 0; t < T * I; ++t) {
      ffn_gate[t] = gelu_tanh(ffn_gate[t]) * ffn_up[t];
    }

    matmul(ffn_gate, lw.ffn_down, ffn_down, T, H, I);
    if (adapter && l < adapter->layers.size() && adapter->layers[l].ffn_down) {
      apply_lora(ffn_gate, *adapter->layers[l].ffn_down, ffn_down, T);
    }

    std::vector<float> post_ffw_w(H, 1.0f);
    if (lw.post_ffw_norm) dequantize(lw.post_ffw_norm->type, lw.post_ffw_norm->data, post_ffw_w.data(), H);

    for (size_t t = 0; t < T; ++t) {
      rms_norm(ffn_down + t * H, post_ffw_w.data(), ffn_down + t * H, H, cfg_.rms_norm_eps);
      vDSP_vadd(cur + t * H, 1, ffn_down + t * H, 1, cur + t * H, 1, H);
    }

    // PLE residual injection
    if ((ple_token_embd_ || compact_ple_.is_loaded) && lw.inp_gate && lw.ple_proj) {
      matmul(cur, lw.inp_gate, ple_gate, T, Dple, H);
      for (size_t t = 0; t < T; ++t) {
        const float* P_l = ple_combined + (t * L + l) * Dple;
        for (size_t d = 0; d < Dple; ++d) {
          ple_gate[t * Dple + d] = gelu_tanh(ple_gate[t * Dple + d]) * P_l[d];
        }
      }
      matmul(ple_gate, lw.ple_proj, ple_out, T, H, Dple);

      std::vector<float> post_ple_w(H, 1.0f);
      if (lw.post_ple_norm) dequantize(lw.post_ple_norm->type, lw.post_ple_norm->data, post_ple_w.data(), H);

      for (size_t t = 0; t < T; ++t) {
        rms_norm(ple_out + t * H, post_ple_w.data(), ple_out + t * H, H, cfg_.rms_norm_eps);
        vDSP_vadd(cur + t * H, 1, ple_out + t * H, 1, cur + t * H, 1, H);
      }
    }

    // Layer output scalar
    if (lw.layer_scale && lw.layer_scale->data) {
      float scale_val = 1.0f;
      dequantize(lw.layer_scale->type, lw.layer_scale->data, &scale_val, 1);
      vDSP_vsmul(cur, 1, &scale_val, cur, 1, T * H);
    }
  }

  // 4. Output projection and logit trimming (only on final token if requested)
  if (compute_logits) {
    const float* last_hidden = cur + (T - 1) * H;
    std::vector<float> final_norm_w(H, 1.0f);
    if (output_norm_) dequantize(output_norm_->type, output_norm_->data, final_norm_w.data(), H);

    std::vector<float> norm_last(H);
    rms_norm(last_hidden, final_norm_w.data(), norm_last.data(), H, cfg_.rms_norm_eps);

    matmul(norm_last.data(), output_weight_, last_logits_.data(), 1, cfg_.vocab_size, H);

    // Final logit softcapping
    float cap = cfg_.logit_softcap;
    if (cap > 0.0f) {
      for (size_t v = 0; v < cfg_.vocab_size; ++v) {
        last_logits_[v] = cap * std::tanh(last_logits_[v] / cap);
      }
    }
  }
}

const float* Gemma4Engine::prefill(uint32_t slot_idx, const std::vector<int32_t>& tokens, bool keep_logits) {
  if (tokens.empty()) return nullptr;
  ensure_model_loaded();
  ensure_kv_loaded(slot_idx);

  size_t N = tokens.size();
  size_t step = cfg_.prefill_step;

  int32_t& pos = (slot_idx < slots_.size()) ? slots_[slot_idx].seq_pos : seq_pos_;

  for (size_t i = 0; i < N; i += step) {
    size_t chunk_len = std::min(step, N - i);
    bool is_last_chunk = (i + chunk_len == N);
    forward_chunk(slot_idx, &tokens[i], chunk_len, pos, is_last_chunk && keep_logits);
    pos += int32_t(chunk_len);
  }

  return keep_logits ? last_logits_.data() : nullptr;
}

const float* Gemma4Engine::decode(uint32_t slot_idx, int32_t token) {
  ensure_model_loaded();
  ensure_kv_loaded(slot_idx);

  int32_t& pos = (slot_idx < slots_.size()) ? slots_[slot_idx].seq_pos : seq_pos_;
  forward_chunk(slot_idx, &token, 1, pos, true);
  pos += 1;
  return last_logits_.data();
}

} // namespace ni
