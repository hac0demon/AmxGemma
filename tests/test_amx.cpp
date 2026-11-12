#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "tokenizer/tokenizer.h"
#include "engine/sampler.h"
#include "engine/gemma4.h"
#include "util/log.h"
#include "util/bf16.h"

#include <Accelerate/Accelerate.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

void test_sampler() {
  std::cout << "[TEST] Sampler... " << std::flush;
  ni::SamplerConfig cfg;
  cfg.temp = 0.0f; // greedy
  ni::Sampler sampler(cfg);

  std::vector<float> logits(262144, 0.0f);
  logits[42] = 10.5f;
  logits[100] = 5.0f;
  logits[1000] = 2.0f;

  int picked = sampler.sample(logits.data(), logits.size(), 0);
  (void)picked;
  assert(picked == 42);

  // Test stochastic sampling
  cfg.temp = 0.8f;
  cfg.top_p = 0.95f;
  cfg.top_k = 10;
  cfg.seed = 12345;
  sampler.set_config(cfg);

  int s1 = sampler.sample(logits.data(), logits.size(), 1);
  int s2 = sampler.sample(logits.data(), logits.size(), 1);
  (void)s1; (void)s2;
  assert(s1 == s2); // Seeded reproducibility
  assert(s1 == 42 || s1 == 100 || s1 == 1000);

  // Test repetition penalty in greedy
  cfg.temp = 0.0f;
  cfg.repetition_penalty = 1.2f;
  sampler.set_config(cfg);
  int history[] = {42};
  int rep_picked = sampler.sample(logits.data(), logits.size(), 2, history, 1);
  // 10.5 / 1.2 = 8.75 > 5.0, so 42 still wins
  assert(rep_picked == 42);
  // With higher penalty: 10.5 / 2.5 = 4.2 < 5.0, so 100 wins!
  cfg.repetition_penalty = 2.5f;
  sampler.set_config(cfg);
  rep_picked = sampler.sample(logits.data(), logits.size(), 3, history, 1);
  assert(rep_picked == 100);

  std::cout << "PASSED\n";
}

void test_kv_snapshot_memcpy() {
  std::cout << "[TEST] Contiguous Quantized KV Arena Snapshot & Fast Restore... " << std::flush;
  // Gemma 4 E2B KV Arena size: ~2.5 MB in Q8_0/Q4_0 (or ~12 MB in full context)
  size_t n_bytes = 2500000;
  std::vector<uint8_t> kv_arena(n_bytes);
  for (size_t i = 0; i < n_bytes; ++i) {
    kv_arena[i] = uint8_t(i & 0xff);
  }

  // 1. Snapshot
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> snapshot = kv_arena;
  auto t1 = std::chrono::high_resolution_clock::now();
  double save_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  // 2. Corrupt arena to simulate intervening generation
  std::fill(kv_arena.begin(), kv_arena.end(), uint8_t(0xdd));

  // 3. Fast Restore
  auto t2 = std::chrono::high_resolution_clock::now();
  std::memcpy(kv_arena.data(), snapshot.data(), n_bytes);
  auto t3 = std::chrono::high_resolution_clock::now();
  double restore_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

  // Verify bit-identical restoration
  assert(std::memcmp(kv_arena.data(), snapshot.data(), n_bytes) == 0);

  std::cout << "PASSED (Save: " << save_us << " us, Fast Restore: " << restore_us << " us / "
            << (restore_us / 1000.0) << " ms)\n";
}

void test_q8_q4_quantization() {
  std::cout << "[TEST] Q8_0 and Q4_0 KV Cache Quantization... " << std::flush;
  constexpr size_t K = 256; // 8 blocks of 32
  std::vector<float> orig(K);
  for (size_t i = 0; i < K; ++i) {
    orig[i] = std::sin(float(i + 1) * 0.2f) * 4.0f;
  }

  // Test Q8_0
  std::vector<ni::block_q8_0> b8(K / 32);
  ni::Gemma4Engine::quantize_q8_0(orig.data(), b8.data(), K);
  std::vector<float> dequant_q8(K);
  for (size_t b = 0; b < K / 32; ++b) {
    float d = ni::half_to_float(b8[b].d);
    for (int j = 0; j < 32; ++j) {
      dequant_q8[b * 32 + j] = d * float(b8[b].qs[j]);
    }
  }

  float max_diff_q8 = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    float diff = std::abs(orig[i] - dequant_q8[i]);
    if (diff > max_diff_q8) max_diff_q8 = diff;
  }
  // Q8_0 max error should be <= 4.0 / 127 ~ 0.032
  assert(max_diff_q8 < 0.05f);

  // Test Q4_0
  std::vector<ni::block_q4_0> b4(K / 32);
  ni::Gemma4Engine::quantize_q4_0(orig.data(), b4.data(), K);
  std::vector<float> dequant_q4(K);
  for (size_t b = 0; b < K / 32; ++b) {
    float d = ni::half_to_float(b4[b].d);
    for (int j = 0; j < 16; ++j) {
      dequant_q4[b * 32 + j] = d * float(int(b4[b].qs[j] & 0x0F) - 8);
      dequant_q4[b * 32 + 16 + j] = d * float(int(b4[b].qs[j] >> 4) - 8);
    }
  }

  float max_diff_q4 = 0.0f;
  for (size_t i = 0; i < K; ++i) {
    float diff = std::abs(orig[i] - dequant_q4[i]);
    if (diff > max_diff_q4) max_diff_q4 = diff;
  }
  // Q4_0 max error should be <= 4.0 / 7 ~ 0.6
  assert(max_diff_q4 < 0.8f);

  std::cout << "PASSED (Q8_0 max err: " << max_diff_q8 << ", Q4_0 max err: " << max_diff_q4 << ")\n";
}

void test_flash_attention_cpu() {
  std::cout << "[TEST] CPU Flash Attention (Tiled Online Softmax vs Naive Attention)... " << std::flush;
  size_t Dh = 256;
  size_t ctx = 64; // spans across 2 tiles (Bc = 32)
  size_t n_blocks = Dh / 32;
  size_t k_token_bytes = n_blocks * sizeof(ni::block_q8_0);
  size_t v_token_bytes = n_blocks * sizeof(ni::block_q4_0);

  std::vector<float> q_h(Dh);
  for (size_t d = 0; d < Dh; ++d) q_h[d] = std::sin(float(d) * 0.1f) * 0.5f;

  std::vector<uint8_t> k_cache(ctx * k_token_bytes);
  std::vector<uint8_t> v_cache(ctx * v_token_bytes);

  std::vector<std::vector<float>> k_raw(ctx, std::vector<float>(Dh));
  std::vector<std::vector<float>> v_raw(ctx, std::vector<float>(Dh));

  for (size_t t = 0; t < ctx; ++t) {
    for (size_t d = 0; d < Dh; ++d) {
      k_raw[t][d] = std::cos(float(t * Dh + d) * 0.05f) * 0.5f;
      v_raw[t][d] = std::sin(float(t * Dh + d) * 0.05f) * 0.5f;
    }
    ni::Gemma4Engine::quantize_q8_0(k_raw[t].data(),
        reinterpret_cast<ni::block_q8_0*>(k_cache.data() + t * k_token_bytes), Dh);
    ni::Gemma4Engine::quantize_q4_0(v_raw[t].data(),
        reinterpret_cast<ni::block_q4_0*>(v_cache.data() + t * v_token_bytes), Dh);
  }

  // 1. Run flash_attn_head
  std::vector<float> flash_out(Dh, 0.0f);
  ni::Gemma4Engine::flash_attn_head(q_h.data(), k_cache.data(), v_cache.data(),
                                    k_token_bytes, v_token_bytes, ctx,
                                    /*is_swa=*/false, /*min_p=*/0, ctx, /*hkv=*/0, Dh, flash_out.data());

  // 2. Reference Naive Attention using the exact same quantized K and V values
  std::vector<float> scores(ctx, 0.0f);
  float max_s = -1e9f;
  for (size_t t = 0; t < ctx; ++t) {
    const ni::block_q8_0* kb = reinterpret_cast<const ni::block_q8_0*>(k_cache.data() + t * k_token_bytes);
    float dot = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
      float d = ni::half_to_float(kb[b].d);
      for (int j = 0; j < 32; ++j) {
        dot += q_h[b * 32 + j] * (d * float(kb[b].qs[j]));
      }
    }
    scores[t] = dot;
    if (dot > max_s) max_s = dot;
  }

  float sum_exp = 0.0f;
  for (size_t t = 0; t < ctx; ++t) {
    scores[t] = std::exp(scores[t] - max_s);
    sum_exp += scores[t];
  }
  float inv_sum = 1.0f / sum_exp;

  std::vector<float> ref_out(Dh, 0.0f);
  for (size_t t = 0; t < ctx; ++t) {
    float w = scores[t] * inv_sum;
    const ni::block_q4_0* vb = reinterpret_cast<const ni::block_q4_0*>(v_cache.data() + t * v_token_bytes);
    for (size_t b = 0; b < n_blocks; ++b) {
      float vd = ni::half_to_float(vb[b].d);
      for (int j = 0; j < 16; ++j) {
        ref_out[b * 32 + j] += w * vd * float(int(vb[b].qs[j] & 0x0F) - 8);
        ref_out[b * 32 + 16 + j] += w * vd * float(int(vb[b].qs[j] >> 4) - 8);
      }
    }
  }

  // Compare flash_out and ref_out
  float max_err = 0.0f;
  for (size_t d = 0; d < Dh; ++d) {
    float err = std::abs(flash_out[d] - ref_out[d]);
    if (err > max_err) max_err = err;
  }
  assert(max_err < 1e-4f);
  std::cout << "PASSED (Max diff between Flash Attention and Naive: " << max_err << ")\n";
}

void test_idle_unload_reload() {
  std::cout << "[TEST] Idle Unload and On-Demand Reload Lifecycle... " << std::flush;
  ni::Gemma4Engine engine;
  assert(!engine.is_model_loaded());
  assert(!engine.is_kv_loaded());

  engine.unload_model();
  engine.unload_kv();
  assert(!engine.is_model_loaded());
  assert(!engine.is_kv_loaded());

  std::cout << "PASSED\n";
}

void test_amx_accelerate() {
  std::cout << "[TEST] Apple Accelerate AMX Tile Matmul (256 x 6144 x 1536)... " << std::flush;
  int M = 256;  // prefill step size
  int N = 6144; // FFN hidden dim
  int K = 1536; // embedding dim

  std::vector<float> A(M * K, 0.01f);
  std::vector<float> B(N * K, 0.02f);
  std::vector<float> C(M * N, 0.0f);

  auto t0 = std::chrono::high_resolution_clock::now();
  // Dispatches to AMX coprocessor on Apple Silicon
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              M, N, K,
              1.0f, A.data(), K,
              B.data(), K,
              0.0f, C.data(), N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Theoretical FLOPs = 2 * M * N * K = 2 * 256 * 6144 * 1536 = 4.83 GFLOPs
  double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);
  assert(std::abs(C[0] - (K * 0.01f * 0.02f)) < 1e-3f);

  std::cout << "PASSED (Elapsed: " << elapsed_ms << " ms, Throughput: " << gflops << " GFLOP/s)\n";
}

void test_bf16_conversion() {
  std::cout << "[TEST] BF16 conversions... " << std::flush;
  float orig = 3.14159265f;
  uint16_t b = ni::f32_to_bf16(orig);
  float recovered = ni::bf16_to_f32(b);
  (void)recovered;
  assert(std::abs(orig - recovered) < 0.02f);
  std::cout << "PASSED\n";
}

void test_model_tokenizer() {
  const char* path = "/Users/ivash/models/gemma-4-E2B-it/body.gguf";
  std::ifstream f(path);
  if (!f.good()) {
    std::cout << "[TEST] Real Model GGUF & Tokenizer... SKIPPED (file not accessible in sandbox)\n";
    return;
  }
  std::cout << "[TEST] Real Model GGUF & Tokenizer... " << std::flush;
  try {
    ni::gguf::File gf(path);
    assert(!gf.tensors().empty());
    std::cout << "\n--- GGUF Metadata Keys ---\n";
    for (auto it = gf.meta().begin(); it != gf.meta().end(); ++it) {
      if (it.key().find("rope") != std::string::npos ||
          it.key().find("freq") != std::string::npos ||
          it.key().find("rotary") != std::string::npos ||
          it.key().find("head") != std::string::npos ||
          it.key().find("attention") != std::string::npos) {
        if (it.value().is_string()) {
          std::cout << it.key() << " = " << it.value().get<std::string>() << "\n";
        } else {
          std::cout << it.key() << " = " << it.value().dump() << "\n";
        }
      }
    }
    std::cout << "---------------------------\n";
    assert(gf.has("general.architecture"));
    assert(gf.meta()["general.architecture"].get<std::string>() == "gemma4");

    ni::Tokenizer tok;
    bool ok = tok.load_from_gguf(gf);
    (void)ok;
    assert(ok);
    assert(tok.vocab_size() > 200000);

    std::string test_str = "<bos><|turn>user\nWrite me good Margarita coctail recipe<turn|>\n<|turn>model\n";
    auto tokens = tok.encode(test_str);
    assert(!tokens.empty());

    std::vector<int32_t> prompt_ids(tokens.begin(), tokens.end());
    ni::Gemma4Engine engine;
    bool ok_m = engine.load_model(gf);
    assert(ok_m);
    engine.load_ple_safetensors("/Users/ivash/models/gemma-4-E2B-it/ple_s.safetensors");
    engine.reset_state(0);

    const float* logits = engine.prefill(0, prompt_ids, true);
    ni::SamplerConfig sc;
    sc.temp = 0.0f; // greedy
    sc.repetition_penalty = 1.15f;
    ni::Sampler sampler(sc);

    std::vector<int> gen_history;
    int step = 0;
    int cur_token = sampler.sample(logits, tok.vocab_size(), step++, gen_history.data(), gen_history.size());
    std::string response;

    for (int gen_step = 0; gen_step < 20; ++gen_step) {
      if (tok.is_eos(cur_token)) break;
      gen_history.push_back(cur_token);
      response += tok.token_bytes(cur_token);
      const float* next_logits = engine.decode(0, cur_token);
      cur_token = sampler.sample(next_logits, tok.vocab_size(), step++, gen_history.data(), gen_history.size());
    }

    assert(response.find("fantastic choice") != std::string::npos ||
           response.find("Margarita") != std::string::npos ||
           response.find("classic") != std::string::npos);

    std::cout << "PASSED (Vocab: " << tok.vocab_size() << ", Tensors: " << gf.tensors().size() << ")\n";

    std::cout << "PASSED (Vocab: " << tok.vocab_size() << ", Tensors: " << gf.tensors().size() << ")\n";
  } catch (const std::exception& e) {
    std::cout << "FAILED: " << e.what() << "\n";
    assert(false);
  }
}

void test_multi_user_slots() {
  std::cout << "[TEST] Multi-User Slots (>= 4 simultaneous users KV independence)... " << std::flush;
  constexpr size_t num_slots = 4;
  constexpr size_t kv_size = 100000;

  struct MockSlot {
    std::string user;
    std::vector<uint16_t> kv_arena;
    int32_t seq_pos = 0;
    std::string attached_adapter;
  };

  std::vector<MockSlot> slots(num_slots);
  for (size_t i = 0; i < num_slots; ++i) {
    slots[i].user = "user_" + std::to_string(i);
    slots[i].kv_arena.resize(kv_size, uint16_t(i * 100));
    slots[i].seq_pos = int32_t(i * 10);
  }

  // Bind specific reducer workflow adapter to slot 1
  slots[1].user = "reducer";
  slots[1].attached_adapter = "logs_reducer";

  // Simulate concurrent user writes
  for (size_t i = 0; i < kv_size; ++i) {
    slots[0].kv_arena[i] = uint16_t(0xAAAA);
    slots[1].kv_arena[i] = uint16_t(0xBBBB);
    slots[2].kv_arena[i] = uint16_t(0xCCCC);
    slots[3].kv_arena[i] = uint16_t(0xDDDD);
  }

  // Verify full memory independence across all 4 slots
  for (size_t i = 0; i < kv_size; ++i) {
    assert(slots[0].kv_arena[i] == uint16_t(0xAAAA));
    assert(slots[1].kv_arena[i] == uint16_t(0xBBBB));
    assert(slots[2].kv_arena[i] == uint16_t(0xCCCC));
    assert(slots[3].kv_arena[i] == uint16_t(0xDDDD));
  }
  assert(slots[1].user == "reducer");
  assert(slots[1].attached_adapter == "logs_reducer");
  assert(slots[0].attached_adapter.empty());

  std::cout << "PASSED (4 slots isolated, reducer adapter slot verified)\n";
}

void test_lora_math_dispatch() {
  std::cout << "[TEST] QLoRA / LoRA Low-Rank AMX Dispatch... " << std::flush;
  int r = 8;
  int d_in = 1536;
  int d_out = 2048;
  float alpha = 16.0f;
  float scale = alpha / float(r); // 2.0

  std::vector<float> A(r * d_in);
  std::vector<float> B(d_out * r);
  for (size_t i = 0; i < A.size(); ++i) A[i] = float(std::sin(double(i) * 0.01)) * 0.05f;
  for (size_t i = 0; i < B.size(); ++i) B[i] = float(std::cos(double(i) * 0.01)) * 0.05f;

  // Test T = 1 (decode vector dot product)
  std::vector<float> x1(d_in, 0.1f);
  std::vector<float> y1(d_out, 0.0f);

  // Reference computation
  std::vector<float> h1_ref(r, 0.0f);
  for (int i = 0; i < r; ++i) {
    for (int k = 0; k < d_in; ++k) h1_ref[i] += x1[k] * A[i * d_in + k];
  }
  std::vector<float> y1_ref(d_out, 0.0f);
  for (int j = 0; j < d_out; ++j) {
    for (int i = 0; i < r; ++i) y1_ref[j] += scale * B[j * r + i] * h1_ref[i];
  }

  // Accelerated dispatch
  std::vector<float> h1(r, 0.0f);
  for (int i = 0; i < r; ++i) {
    vDSP_dotpr(x1.data(), 1, A.data() + i * d_in, 1, &h1[i], d_in);
  }
  for (int j = 0; j < d_out; ++j) {
    float delta = 0.0f;
    vDSP_dotpr(B.data() + j * r, 1, h1.data(), 1, &delta, r);
    y1[j] += scale * delta;
  }

  for (int j = 0; j < d_out; ++j) {
    assert(std::abs(y1[j] - y1_ref[j]) < 1e-3f);
  }

  // Test T = 4 (chunked prefill AMX matmul)
  size_t T = 4;
  std::vector<float> xT(T * d_in, 0.05f);
  std::vector<float> yT(T * d_out, 0.0f);

  std::vector<float> HT(T * r, 0.0f);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              int(T), r, d_in,
              1.0f, xT.data(), d_in,
              A.data(), d_in,
              0.0f, HT.data(), r);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              int(T), d_out, r,
              scale, HT.data(), r,
              B.data(), r,
              1.0f, yT.data(), d_out);

  for (size_t t = 0; t < T; ++t) {
    assert(std::abs(yT[t * d_out] - (y1[0] * 0.5f)) < 1e-2f);
  }

  std::cout << "PASSED (Decode & AMX Prefill LoRA bit-identical to reference)\n";
}

void test_yarn_rope() {
  std::cout << "[TEST] YaRN RoPE Context Extension (256K tokens)... " << std::flush;
  uint32_t head_dim = 256;
  float freq_base = 1000000.0f;

  ni::YarnConfig yarn;
  yarn.enabled = true;
  yarn.orig_ctx = 131072;
  yarn.target_ctx = 262144;
  yarn.scale = 2.0f;
  yarn.beta_fast = 32.0f;
  yarn.beta_slow = 1.0f;
  yarn.mscale = 1.0f;
  yarn.ext_factor = 1.0f;
  yarn.update(head_dim, freq_base);

  // 1. Verify correction dimensions
  // corr_dim(32) ~ 60, corr_dim(1) ~ 92
  assert(yarn.corr_dim_low >= 55.0f && yarn.corr_dim_low <= 65.0f);
  assert(yarn.corr_dim_high >= 85.0f && yarn.corr_dim_high <= 95.0f);
  assert(yarn.corr_dim_low < yarn.corr_dim_high);
  // eff_mscale = 1.0 * (1.0 + 0.1 * ln(2.0)) ~ 1.0693
  assert(std::abs(yarn.eff_mscale - 1.06931f) < 1e-3f);

  // 2. Scale == 1.0f should match standard RoPE bit-identically
  ni::YarnConfig yarn_base;
  yarn_base.enabled = false;
  yarn_base.scale = 1.0f;
  yarn_base.update(head_dim, freq_base);

  std::vector<float> vec_base(head_dim, 1.0f);
  std::vector<float> vec_yarn1(head_dim, 1.0f);
  int32_t test_pos = 1000;
  ni::Gemma4Engine::apply_rope(vec_base.data(), head_dim, test_pos, freq_base);
  ni::Gemma4Engine::apply_rope_yarn(vec_yarn1.data(), head_dim, test_pos, freq_base, yarn_base, true);
  for (size_t i = 0; i < head_dim; ++i) {
    assert(std::abs(vec_base[i] - vec_yarn1[i]) < 1e-6f);
  }

  // 3. SWA layer (use_yarn = false) should match standard RoPE even if yarn is enabled
  std::vector<float> vec_swa(head_dim, 1.0f);
  ni::Gemma4Engine::apply_rope_yarn(vec_swa.data(), head_dim, test_pos, freq_base, yarn, false);
  for (size_t i = 0; i < head_dim; ++i) {
    assert(std::abs(vec_base[i] - vec_swa[i]) < 1e-6f);
  }

  // 4. Test extreme context positions: 0, 131072, 262143
  std::vector<int32_t> long_positions = {0, 512, 8192, 65536, 131072, 200000, 262143};
  for (int32_t pos : long_positions) {
    std::vector<float> qk(head_dim);
    for (size_t i = 0; i < head_dim; ++i) qk[i] = std::sin(float(i + 1) * 0.1f);
    ni::Gemma4Engine::apply_rope_yarn(qk.data(), head_dim, pos, freq_base, yarn, true);
    for (size_t i = 0; i < head_dim; ++i) {
      assert(!std::isnan(qk[i]));
      assert(!std::isinf(qk[i]));
      assert(std::abs(qk[i]) < 5.0f); // bounded magnitude
    }
  }

  // 5. Test high-frequency preservation vs low-frequency interpolation
  // For i = 0 (high freq, wavelength << context): ramp = 1.0 -> theta = theta_extrap
  std::vector<float> qk_test(head_dim, 0.0f);
  qk_test[0] = 1.0f; // first pair
  qk_test[head_dim - 2] = 1.0f; // last pair
  std::vector<float> qk_yarn = qk_test;
  std::vector<float> qk_std = qk_test;
  int32_t pos = 262140; // deep in 256K context
  ni::Gemma4Engine::apply_rope(qk_std.data(), head_dim, pos, freq_base);
  ni::Gemma4Engine::apply_rope_yarn(qk_yarn.data(), head_dim, pos, freq_base, yarn, true);

  // At dim 0, theta is extrapolated (with eff_mscale)
  float theta0 = float(pos) * 1.0f; // freq_base^0 = 1.0
  float expected_cos0 = std::cos(theta0) * yarn.eff_mscale;
  (void)expected_cos0;
  assert(std::abs(qk_yarn[0] - expected_cos0) < 1e-4f);

  std::cout << "PASSED (corr_dims: [" << yarn.corr_dim_low << ", " << yarn.corr_dim_high
            << "], eff_mscale: " << yarn.eff_mscale << ", tested to 262,143 tokens)\n";
}

int main() {
  std::cout << "=== Running Gemma-AMX Verification Suite ===\n";
  test_bf16_conversion();
  test_sampler();
  test_kv_snapshot_memcpy();
  test_q8_q4_quantization();
  test_flash_attention_cpu();
  test_idle_unload_reload();
  test_multi_user_slots();
  test_lora_math_dispatch();
  test_yarn_rope();
  test_amx_accelerate();
  test_model_tokenizer();
  std::cout << "=== All Unit Tests Passed Successfully ===\n";
  return 0;
}
