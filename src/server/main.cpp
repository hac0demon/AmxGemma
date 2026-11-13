#include "gguf/gguf.h"
#include "tokenizer/tokenizer.h"
#include "engine/gemma4.h"
#include "engine/sampler.h"
#include "util/log.h"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>
#include <minja/chat-template.hpp>

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using ni::logi;
using ni::loge;

namespace {

std::unique_ptr<httplib::Server> g_server;
std::mutex g_engine_mutex;
std::atomic<bool> g_running{true};
std::atomic<int64_t> g_last_request_time_ms{0};

void sig_handler(int) {
  g_running = false;
  if (g_server) g_server->stop();
}

std::string read_file_string(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string format_gemma4_messages(const nlohmann::json& messages, bool add_generation_prompt, bool skip_first_system = false) {
  std::string out;
  if (!skip_first_system) out += "<bos>";
  bool first = true;
  for (const auto& msg : messages) {
    std::string role = msg.value("role", "user");
    std::string content = msg.value("content", "");
    if (first && skip_first_system && role == "system") {
      first = false;
      continue;
    }
    first = false;
    out += "<|turn>" + role + "\n" + content + "<turn|>\n";
  }
  if (add_generation_prompt) {
    out += "<|turn>model\n";
  }
  return out;
}

// ---------------------------------------------------------------------------
// Slot Manager (Thread-safe routing for simultaneous access by >= 4 users)
// ---------------------------------------------------------------------------

class SlotManager {
 public:
  SlotManager(ni::Gemma4Engine& engine, size_t num_slots = 4)
      : engine_(engine), num_slots_(num_slots) {
    engine_.init_slots(uint32_t(num_slots));
  }

  int acquire_slot(const std::string& user) {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t now = int64_t(ni::now_ms());

    // 1. If user is non-empty, look for slot already bound to this user
    if (!user.empty()) {
      for (size_t i = 0; i < num_slots_; ++i) {
        if (engine_.slot(uint32_t(i)).bound_user == user) {
          engine_.slot(uint32_t(i)).last_accessed_ms = now;
          return int(i);
        }
      }
    }

    // 2. Look for an unbound slot that is not currently active
    for (size_t i = 0; i < num_slots_; ++i) {
      if (engine_.slot(uint32_t(i)).bound_user.empty() && !engine_.slot(uint32_t(i)).is_active) {
        engine_.slot(uint32_t(i)).bound_user = user;
        engine_.slot(uint32_t(i)).last_accessed_ms = now;
        return int(i);
      }
    }

    // 3. Look for the LRU non-active slot
    int best_slot = -1;
    int64_t oldest = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < num_slots_; ++i) {
      if (!engine_.slot(uint32_t(i)).is_active && engine_.slot(uint32_t(i)).last_accessed_ms < oldest) {
        oldest = engine_.slot(uint32_t(i)).last_accessed_ms;
        best_slot = int(i);
      }
    }

    if (best_slot < 0) {
      best_slot = 0;
    }

    engine_.slot(uint32_t(best_slot)).bound_user = user;
    engine_.slot(uint32_t(best_slot)).last_accessed_ms = now;
    return best_slot;
  }

  void set_active(int slot_idx, bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx >= 0 && slot_idx < int(num_slots_)) {
      engine_.slot(uint32_t(slot_idx)).is_active = active;
      engine_.slot(uint32_t(slot_idx)).last_accessed_ms = int64_t(ni::now_ms());
    }
  }

  bool attach_adapter(const std::string& target, const std::string& adapter_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    char* end = nullptr;
    long s_idx = strtol(target.c_str(), &end, 10);
    if (*end == '\0' && s_idx >= 0 && s_idx < long(num_slots_)) {
      return engine_.attach_adapter_to_slot(uint32_t(s_idx), adapter_id);
    }
    for (size_t i = 0; i < num_slots_; ++i) {
      if (engine_.slot(uint32_t(i)).bound_user == target) {
        return engine_.attach_adapter_to_slot(uint32_t(i), adapter_id);
      }
    }
    for (size_t i = 0; i < num_slots_; ++i) {
      if (engine_.slot(uint32_t(i)).bound_user.empty()) {
        engine_.slot(uint32_t(i)).bound_user = target;
        return engine_.attach_adapter_to_slot(uint32_t(i), adapter_id);
      }
    }
    return engine_.attach_adapter_to_slot(0, adapter_id);
  }

  nlohmann::json get_slots_info() {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json arr = nlohmann::json::array();
    for (size_t i = 0; i < num_slots_; ++i) {
      const auto& s = engine_.slot(uint32_t(i));
      arr.push_back({
        {"slot_id", s.slot_id},
        {"user", s.bound_user},
        {"seq_pos", s.seq_pos},
        {"is_active", s.is_active},
        {"is_primed", s.snapshot.is_primed},
        {"adapter", s.attached_adapter_id},
        {"last_accessed_ms", s.last_accessed_ms}
      });
    }
    return arr;
  }

  size_t num_slots() const { return num_slots_; }

 private:
  ni::Gemma4Engine& engine_;
  size_t num_slots_;
  std::mutex mutex_;
};

} // namespace

uint32_t parse_ctx_size(const std::string& str) {
  std::string s = str;
  for (char& c : s) c = char(std::tolower(c));
  size_t mult = 1;
  if (!s.empty() && s.back() == 'k') {
    mult = 1024;
    s.pop_back();
  } else if (!s.empty() && s.back() == 'm') {
    mult = 1024 * 1024;
    s.pop_back();
  }
  return uint32_t(std::stoul(s) * mult);
}

int main(int argc, char** argv) {
  std::string model_path;
  std::string ple_path;
  std::string host = "127.0.0.1";
  int port = 8081;
  uint32_t ctx = 8192;
  uint32_t num_slots = 4;
  std::string static_system_prompt;
  std::string test_prompt;
  int max_tokens = 2048;
  float temp = 0.0f;
  float top_p = 0.95f;
  uint32_t top_k = 40;
  float repetition_penalty = 1.15f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  bool preload_amx = false;
  int idle_timeout_model_sec = 0;
  int idle_timeout_kv_sec = 0;
  bool no_fa = false;
  std::vector<std::string> adapter_args;
  std::vector<std::string> slot_adapter_args;

  // YaRN RoPE context extension parameters
  bool yarn_enabled = false;
  bool yarn_set_explicitly = false;
  uint32_t yarn_orig_ctx = 131072;
  float yarn_scale = 0.0f;
  float yarn_beta_fast = 32.0f;
  float yarn_beta_slow = 1.0f;
  float yarn_mscale = 1.0f;
  float yarn_ext_factor = 1.0f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
    else if (arg == "--ple" && i + 1 < argc) ple_path = argv[++i];
    else if (arg == "--host" && i + 1 < argc) host = argv[++i];
    else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (arg == "--ctx" && i + 1 < argc) ctx = parse_ctx_size(argv[++i]);
    else if (arg == "--slots" && i + 1 < argc) num_slots = uint32_t(std::stoi(argv[++i]));
    else if (arg == "--idle-timeout-model" && i + 1 < argc) idle_timeout_model_sec = std::stoi(argv[++i]);
    else if (arg == "--idle-timeout-kv" && i + 1 < argc) idle_timeout_kv_sec = std::stoi(argv[++i]);
    else if (arg == "--idle-timeout" && i + 1 < argc) {
      idle_timeout_model_sec = std::stoi(argv[++i]);
      idle_timeout_kv_sec = idle_timeout_model_sec * 2;
    }
    else if (arg == "--no-fa") no_fa = true;
    else if (arg == "--yarn") { yarn_enabled = true; yarn_set_explicitly = true; }
    else if (arg == "--yarn-orig-ctx" && i + 1 < argc) yarn_orig_ctx = parse_ctx_size(argv[++i]);
    else if (arg == "--yarn-scale" && i + 1 < argc) yarn_scale = std::stof(argv[++i]);
    else if (arg == "--yarn-beta-fast" && i + 1 < argc) yarn_beta_fast = std::stof(argv[++i]);
    else if (arg == "--yarn-beta-slow" && i + 1 < argc) yarn_beta_slow = std::stof(argv[++i]);
    else if (arg == "--yarn-mscale" && i + 1 < argc) yarn_mscale = std::stof(argv[++i]);
    else if (arg == "--yarn-ext-factor" && i + 1 < argc) yarn_ext_factor = std::stof(argv[++i]);
    else if (arg == "--adapter" && i + 1 < argc) adapter_args.push_back(argv[++i]);
    else if (arg == "--slot-adapter" && i + 1 < argc) slot_adapter_args.push_back(argv[++i]);
    else if (arg == "--system-prompt" && i + 1 < argc) static_system_prompt = argv[++i];
    else if (arg == "--system-prompt-file" && i + 1 < argc) static_system_prompt = read_file_string(argv[++i]);
    else if (arg == "--prompt" && i + 1 < argc) test_prompt = argv[++i];
    else if ((arg == "--max-tokens" || arg == "-n" || arg == "--n-predict") && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
    else if (arg == "--temp" && i + 1 < argc) temp = std::stof(argv[++i]);
    else if (arg == "--top-p" && i + 1 < argc) top_p = std::stof(argv[++i]);
    else if (arg == "--top-k" && i + 1 < argc) top_k = uint32_t(std::stoi(argv[++i]));
    else if ((arg == "--rep-penalty" || arg == "--repetition-penalty") && i + 1 < argc) repetition_penalty = std::stof(argv[++i]);
    else if (arg == "--presence-penalty" && i + 1 < argc) presence_penalty = std::stof(argv[++i]);
    else if (arg == "--frequency-penalty" && i + 1 < argc) frequency_penalty = std::stof(argv[++i]);
    else if (arg == "--preload-amx") preload_amx = true;
    else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: gemma-amx --model <model.gguf> [options]\n"
                << "Options:\n"
                << "  --port <port>                 Server port (default: 8081)\n"
                << "  --host <host>                 Server host (default: 127.0.0.1)\n"
                << "  --ctx <num>                   Context size in tokens (e.g. 256k, 128k, 8192)\n"
                << "  --slots <num>                 Simultaneous user slots (default: 4)\n"
                << "  --idle-timeout-model <sec>    Unload model weights after idle seconds (default: 0 = disabled)\n"
                << "  --idle-timeout-kv <sec>       Unload slot KV caches after idle seconds (default: 0 = disabled)\n"
                << "  --idle-timeout <sec>          Set model timeout, and 2x for KV timeout\n"
                << "  --no-fa                       Disable Flash Attention (default: enabled)\n"
                << "  --yarn                        Force-enable YaRN context extension\n"
                << "  --yarn-orig-ctx <num>         YaRN original base context (default: 131072)\n"
                << "  --yarn-scale <float>          YaRN scale factor override (default: target/orig)\n"
                << "  --yarn-beta-fast <float>      YaRN beta_fast high-frequency boundary (default: 32.0)\n"
                << "  --yarn-beta-slow <float>      YaRN beta_slow low-frequency boundary (default: 1.0)\n"
                << "  --yarn-mscale <float>         YaRN attention scale multiplier (default: 1.0)\n"
                << "  --adapter <name>:<path>       Load LoRA adapter from GGUF/Safetensors\n"
                << "  --slot-adapter <target>:<name> Attach adapter to slot index or user\n"
                << "  --system-prompt <str>         Static system prompt for fast KV snapshot\n"
                << "  --system-prompt-file <path>   File containing static system prompt\n"
                << "  --prompt <str>                One-shot CLI generation and benchmark\n"
                << "  --max-tokens, -n <num>        Maximum generation tokens (default: 2048)\n"
                << "  --temp <float>                Sampling temperature (0.0 = greedy, default: 0.0)\n"
                << "  --top-p <float>               Top-p sampling threshold (default: 0.95)\n"
                << "  --top-k <num>                 Top-k candidates (default: 40)\n"
                << "  --preload-amx                 Pre-convert decoder weights to FP32 for pure AMX\n";
      return 0;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model <path.gguf> is required.\n";
    return 1;
  }

  logi("gemma-amx: starting sidecar on Apple AMX / Accelerate (zero-GPU)");
  double t_boot_start = ni::now_ms();

  // 1. Open GGUF with zero-copy mmap
  std::unique_ptr<ni::gguf::File> gguf_file;
  try {
    gguf_file = std::make_unique<ni::gguf::File>(model_path);
  } catch (const std::exception& e) {
    loge("Failed to open GGUF model %s: %s", model_path.c_str(), e.what());
    return 1;
  }

  // 2. Load Tokenizer directly from GGUF metadata
  ni::Tokenizer tokenizer;
  if (!tokenizer.load_from_gguf(*gguf_file)) {
    loge("Failed to load tokenizer from GGUF metadata");
    return 1;
  }

  // 3. Load Gemma 4 Engine
  ni::Gemma4Engine engine;
  engine.set_flash_attn(!no_fa);
  if (!engine.load_model(*gguf_file, ctx, preload_amx)) {
    loge("Failed to initialize Gemma 4 AMX engine");
    return 1;
  }
  engine.set_model_paths(model_path, ple_path);

  // Configure YaRN context extension if requested or explicitly overridden
  ni::YarnConfig yc;
  yc.orig_ctx = yarn_orig_ctx;
  yc.target_ctx = ctx;
  yc.beta_fast = yarn_beta_fast;
  yc.beta_slow = yarn_beta_slow;
  yc.mscale = yarn_mscale;
  yc.ext_factor = yarn_ext_factor;
  if (yarn_scale > 0.0f) {
    yc.scale = yarn_scale;
  } else {
    yc.scale = float(ctx) / float(yarn_orig_ctx);
  }
  yc.enabled = yarn_set_explicitly ? yarn_enabled : (ctx > yarn_orig_ctx);
  if (yc.enabled || yarn_set_explicitly) {
    engine.set_yarn_config(yc);
  }

  // Auto-detect and load compact PLE safetensors if needed
  if (ple_path.empty()) {
    size_t last_slash = model_path.find_last_of("/\\");
    std::string dir = (last_slash != std::string::npos) ? model_path.substr(0, last_slash) : ".";
    for (const char* name : {"/ple_s.safetensors", "/ple_m.safetensors", "/ple_l.safetensors", "/ple.safetensors"}) {
      std::string candidate = dir + name;
      std::ifstream f(candidate);
      if (f.good()) {
        ple_path = candidate;
        break;
      }
    }
  }
  if (!ple_path.empty()) {
    engine.load_ple_safetensors(ple_path);
    engine.set_model_paths(model_path, ple_path);
  }

  // 4. Initialize Multi-User Slots (>= 4 simultaneous users)
  engine.init_slots(num_slots);
  SlotManager slot_mgr(engine, num_slots);

  // 5. Load and attach configured LoRA / QLoRA adapters
  for (const auto& a : adapter_args) {
    size_t colon = a.find(':');
    if (colon != std::string::npos) {
      std::string id = a.substr(0, colon);
      std::string path = a.substr(colon + 1);
      engine.load_lora_adapter(id, path);
    }
  }

  for (const auto& sa : slot_adapter_args) {
    size_t colon = sa.find(':');
    if (colon != std::string::npos) {
      std::string target = sa.substr(0, colon);
      std::string adapter_id = sa.substr(colon + 1);
      slot_mgr.attach_adapter(target, adapter_id);
    }
  }

  double t_boot_end = ni::now_ms();
  logi("gemma-amx: boot ready in %.1f ms (vocab: %d, ctx: %u, slots: %u, adapters: %zu)",
       t_boot_end - t_boot_start, tokenizer.vocab_size(), ctx, num_slots, engine.list_adapters().size());

  // 6. Prime static system prompt if provided
  if (!static_system_prompt.empty()) {
    std::string sys_turn = "<bos><|turn>system\n" + static_system_prompt + "<turn|>\n";
    std::vector<int> sys_tokens = tokenizer.encode(sys_turn);
    std::vector<int32_t> sys_ids(sys_tokens.begin(), sys_tokens.end());
    engine.prime_system_prompt(sys_ids, 0);
  }

  // 7. One-shot CLI testing path
  if (!test_prompt.empty()) {
    logi("CLI prompt: \"%s\"", test_prompt.c_str());
    std::string formatted;
    if (engine.is_primed(0)) {
      formatted = "<|turn>user\n" + test_prompt + "<turn|>\n<|turn>model\n";
    } else {
      formatted = "<bos><|turn>user\n" + test_prompt + "<turn|>\n<|turn>model\n";
    }

    std::vector<int> prompt_tokens = tokenizer.encode(formatted);
    if (engine.is_primed(0) && !prompt_tokens.empty() && prompt_tokens.front() == tokenizer.bos_id()) {
      prompt_tokens.erase(prompt_tokens.begin());
    }
    std::vector<int32_t> prompt_ids(prompt_tokens.begin(), prompt_tokens.end());

    double t_prefill_start = ni::now_ms();
    if (engine.is_primed(0)) {
      double t_res_start = ni::now_ms();
      engine.restore_snapshot(0);
      logi("Fast restored KV snapshot in %.2f ms (primer: %zu tokens)",
           ni::now_ms() - t_res_start, engine.snapshot(0).primer_ids.size());
    } else {
      engine.reset_state(0);
    }
    const float* logits = engine.prefill(0, prompt_ids, true);
    double t_prefill_end = ni::now_ms();

    double prefill_ms = t_prefill_end - t_prefill_start;
    double prefill_tps = (double(prompt_ids.size()) / (prefill_ms / 1000.0));
    logi("Prefill: %zu tokens in %.1f ms (%.1f tok/s)", prompt_ids.size(), prefill_ms, prefill_tps);

    ni::SamplerConfig sc;
    sc.temp = temp;
    sc.top_p = top_p;
    sc.top_k = top_k;
    sc.repetition_penalty = repetition_penalty;
    sc.presence_penalty = presence_penalty;
    sc.frequency_penalty = frequency_penalty;
    ni::Sampler sampler(sc);

    ni::StreamDecoder decoder(tokenizer);
    std::vector<int> gen_history;
    int step = 0;
    int next_token = sampler.sample(logits, tokenizer.vocab_size(), step++, gen_history.data(), gen_history.size());

    std::cout << "\nOutput:\n";
    double t_decode_start = ni::now_ms();
    int gen_tokens = 0;

    while (gen_tokens < max_tokens && !tokenizer.is_eos(next_token)) {
      gen_history.push_back(next_token);
      std::string chunk = decoder.push(next_token);
      std::cout << chunk << std::flush;
      gen_tokens++;

      const float* next_logits = engine.decode(0, next_token);
      next_token = sampler.sample(next_logits, tokenizer.vocab_size(), step++, gen_history.data(), gen_history.size());
    }
    std::cout << decoder.flush() << "\n\n";

    double decode_ms = ni::now_ms() - t_decode_start;
    double decode_tps = (double(gen_tokens) / (decode_ms / 1000.0));
    logi("Decode: %d tokens in %.1f ms (%.1f tok/s)", gen_tokens, decode_ms, decode_tps);
    return 0;
  }

  // 8. HTTP Sidecar Server
  g_server = std::make_unique<httplib::Server>();
  std::signal(SIGINT, sig_handler);
  std::signal(SIGTERM, sig_handler);
  g_last_request_time_ms = int64_t(ni::now_ms());

  // Health endpoint
  g_server->Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    const auto& ycfg = engine.yarn_config();
    const auto& ecfg = engine.config();
    json j = {
      {"status", "ok"},
      {"model", "gemma-4-e2b"},
      {"model_loaded", engine.is_model_loaded()},
      {"kv_loaded", engine.is_kv_loaded()},
      {"flash_attn", engine.flash_attn_enabled()},
      {"kv_cache", {
        {"k_type", ecfg.kv_type_k},
        {"v_type", ecfg.kv_type_v}
      }},
      {"idle_timeout", {
        {"model_sec", idle_timeout_model_sec},
        {"kv_sec", idle_timeout_kv_sec}
      }},
      {"slots_total", slot_mgr.num_slots()},
      {"primed", engine.is_primed(0)},
      {"primed_tokens", engine.snapshot(0).primer_ids.size()},
      {"context_max", ctx},
      {"adapters", engine.list_adapters()},
      {"yarn", {
        {"enabled", ycfg.enabled},
        {"orig_ctx", ycfg.orig_ctx},
        {"target_ctx", ycfg.target_ctx},
        {"scale", ycfg.scale},
        {"beta_fast", ycfg.beta_fast},
        {"beta_slow", ycfg.beta_slow},
        {"eff_mscale", ycfg.eff_mscale}
      }}
    };
    res.set_content(j.dump(), "application/json");
  });

  // Manual Unload endpoint (force-free model weights, KV caches, or both)
  g_server->Post("/v1/unload", [&](const httplib::Request& req, httplib::Response& res) {
    std::string target = "all";
    if (!req.body.empty()) {
      try {
        json body = json::parse(req.body);
        target = body.value("target", "all");
      } catch (...) {}
    }
    {
      std::lock_guard<std::mutex> lock(g_engine_mutex);
      if (target == "model" || target == "all") {
        engine.unload_model();
      }
      if (target == "kv" || target == "all") {
        engine.unload_kv();
      }
    }
    json resp = {
      {"status", "unloaded"},
      {"target", target},
      {"model_loaded", engine.is_model_loaded()},
      {"kv_loaded", engine.is_kv_loaded()}
    };
    res.set_content(resp.dump(), "application/json");
  });

  // Models endpoint
  g_server->Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json arr = json::array({
      {
        {"id", "gemma-4-e2b"},
        {"object", "model"},
        {"created", 1700000000},
        {"owned_by", "gemma-amx"}
      }
    });
    for (const auto& aid : engine.list_adapters()) {
      arr.push_back({
        {"id", aid},
        {"object", "model"},
        {"created", 1700000000},
        {"owned_by", "lora_adapter"}
      });
    }
    json j = {
      {"object", "list"},
      {"data", arr}
    };
    res.set_content(j.dump(), "application/json");
  });

  // User Slots management endpoint
  g_server->Get("/v1/slots", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(slot_mgr.get_slots_info().dump(), "application/json");
  });

  // Attach adapter to a slot or user endpoint
  g_server->Post("/v1/slots/attach", [&](const httplib::Request& req, httplib::Response& res) {
    g_last_request_time_ms = int64_t(ni::now_ms());
    try {
      json body = json::parse(req.body);
      std::string target;
      if (body.contains("user")) target = body["user"].get<std::string>();
      else if (body.contains("slot")) target = std::to_string(body["slot"].get<int>());
      std::string adapter_id = body.value("adapter", "");

      if (target.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"'user' or 'slot' required\"}", "application/json");
        return;
      }

      bool ok = slot_mgr.attach_adapter(target, adapter_id);
      if (ok) {
        json resp = {{"status", "ok"}, {"target", target}, {"adapter", adapter_id}};
        res.set_content(resp.dump(), "application/json");
      } else {
        res.status = 400;
        res.set_content("{\"error\":\"failed to attach adapter\"}", "application/json");
      }
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  // Adapters list endpoint
  g_server->Get("/v1/adapters", [&](const httplib::Request&, httplib::Response& res) {
    json arr = json::array();
    for (const auto& id : engine.list_adapters()) {
      const auto* a = engine.get_adapter(id);
      arr.push_back({
        {"id", id},
        {"layers", a ? a->layers.size() : 0},
        {"alpha", a ? a->default_alpha : 16.0f}
      });
    }
    res.set_content(arr.dump(), "application/json");
  });

  // Load LoRA adapter dynamically endpoint
  g_server->Post("/v1/adapters/load", [&](const httplib::Request& req, httplib::Response& res) {
    g_last_request_time_ms = int64_t(ni::now_ms());
    try {
      json body = json::parse(req.body);
      std::string id = body.value("id", "");
      std::string path = body.value("path", "");
      float alpha = body.value("alpha", 16.0f);
      if (id.empty() || path.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"id and path required\"}", "application/json");
        return;
      }
      bool ok = engine.load_lora_adapter(id, path, alpha);
      if (ok) {
        json resp = {{"status", "loaded"}, {"id", id}, {"path", path}};
        res.set_content(resp.dump(), "application/json");
      } else {
        res.status = 400;
        res.set_content("{\"error\":\"failed to load adapter from path\"}", "application/json");
      }
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  // Prime endpoint (dynamically update snapshot)
  g_server->Post("/v1/prime", [&](const httplib::Request& req, httplib::Response& res) {
    g_last_request_time_ms = int64_t(ni::now_ms());
    try {
      json body = json::parse(req.body);
      std::string prompt = body.value("system_prompt", "");
      if (prompt.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"system_prompt required\"}", "application/json");
        return;
      }
      static_system_prompt = prompt;
      std::string sys_turn = "<bos><|turn>system\n" + static_system_prompt + "<turn|>\n";
      std::vector<int> sys_tokens = tokenizer.encode(sys_turn);
      std::vector<int32_t> sys_ids(sys_tokens.begin(), sys_tokens.end());

      {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        engine.prime_system_prompt(sys_ids, 0);
      }

      json resp = {
        {"status", "primed"},
        {"tokens", sys_ids.size()}
      };
      res.set_content(resp.dump(), "application/json");
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  // OpenAI Chat Completions endpoint (Concurrent multi-user with slot routing & LoRA)
  g_server->Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
    g_last_request_time_ms = int64_t(ni::now_ms());
    try {
      json body = json::parse(req.body);
      if (!body.contains("messages") || !body["messages"].is_array()) {
        res.status = 400;
        res.set_content("{\"error\":\"messages array required\"}", "application/json");
        return;
      }

      const auto& messages = body["messages"];
      bool stream = body.value("stream", false);
      int req_max_tokens = body.value("max_tokens", max_tokens);
      float req_temp = body.value("temperature", temp);
      float req_top_p = body.value("top_p", top_p);
      uint32_t req_top_k = uint32_t(body.value("top_k", int(top_k)));
      float req_rep_penalty = body.value("repetition_penalty", body.value("rep_penalty", repetition_penalty));
      float req_presence_penalty = body.value("presence_penalty", presence_penalty);
      float req_frequency_penalty = body.value("frequency_penalty", frequency_penalty);
      std::string req_user = body.value("user", "");

      // 1. Acquire dedicated User Slot (supports >= 4 simultaneous users)
      int slot_idx = slot_mgr.acquire_slot(req_user);
      slot_mgr.set_active(slot_idx, true);

      // 2. Check for LoRA adapter attached to slot, or requested via "model"/"adapter"
      std::string req_adapter = body.value("adapter", "");
      if (req_adapter.empty() && body.contains("model")) {
        std::string m = body["model"].get<std::string>();
        if (m != "gemma-4-e2b" && engine.get_adapter(m)) {
          req_adapter = m;
        }
      }
      if (!req_adapter.empty()) {
        engine.attach_adapter_to_slot(uint32_t(slot_idx), req_adapter);
      }

      std::string active_adapter = engine.slot(uint32_t(slot_idx)).attached_adapter_id;
      if (!req_user.empty() || !active_adapter.empty()) {
        logi("chat completion: user='%s', slot=%d, adapter='%s'",
             req_user.c_str(), slot_idx, active_adapter.empty() ? "none" : active_adapter.c_str());
      }

      // 3. Fast restore decision:
      bool use_snapshot = false;
      bool has_system = false;
      if (!messages.empty() && messages[0].value("role", "") == "system") {
        has_system = true;
        if (engine.is_primed(uint32_t(slot_idx)) && messages[0].value("content", "") == static_system_prompt) {
          use_snapshot = true;
        }
      } else if (engine.is_primed(uint32_t(slot_idx))) {
        use_snapshot = true;
      }

      std::string prompt_text;
      {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        if (use_snapshot) {
          engine.restore_snapshot(uint32_t(slot_idx));
          prompt_text = format_gemma4_messages(messages, true, has_system);
        } else {
          engine.reset_state(uint32_t(slot_idx));
          prompt_text = format_gemma4_messages(messages, true, false);
        }
      }

      std::vector<int> prompt_tokens = tokenizer.encode(prompt_text);
      if (use_snapshot && !prompt_tokens.empty() && prompt_tokens.front() == tokenizer.bos_id()) {
        prompt_tokens.erase(prompt_tokens.begin());
      }
      std::vector<int32_t> prompt_ids(prompt_tokens.begin(), prompt_tokens.end());

      // Prefill execution under sequential engine lock
      double t_prefill_start = ni::now_ms();
      const float* logits = nullptr;
      {
        std::lock_guard<std::mutex> lock(g_engine_mutex);
        logits = engine.prefill(uint32_t(slot_idx), prompt_ids, true);
      }
      double prefill_ms = ni::now_ms() - t_prefill_start;
      double prefill_tps = (prefill_ms > 0) ? (double(prompt_ids.size()) / (prefill_ms / 1000.0)) : 0.0;

      if (!logits) {
        slot_mgr.set_active(slot_idx, false);
        res.status = 500;
        res.set_content("{\"error\":\"prefill returned null\"}", "application/json");
        return;
      }

      ni::SamplerConfig sc;
      sc.temp = req_temp;
      sc.top_p = req_top_p;
      sc.top_k = req_top_k;
      sc.repetition_penalty = req_rep_penalty;
      sc.presence_penalty = req_presence_penalty;
      sc.frequency_penalty = req_frequency_penalty;
      ni::Sampler sampler(sc);

      std::string completion_id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

      if (stream) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider("text/event-stream",
          [=, engine_ptr = &engine, tok_ptr = &tokenizer, slot_ptr = &slot_mgr](size_t, httplib::DataSink& sink) mutable -> bool {
            ni::StreamDecoder decoder(*tok_ptr);
            std::vector<int> gen_history;
            int step = 0;
            int next_token = 0;
            double t_decode_start = ni::now_ms();
            {
              std::lock_guard<std::mutex> lock(g_engine_mutex);
              next_token = sampler.sample(logits, tok_ptr->vocab_size(), step++, gen_history.data(), gen_history.size());
            }
            int count = 0;

            while (count < req_max_tokens && !tok_ptr->is_eos(next_token)) {
              gen_history.push_back(next_token);
              std::string delta = decoder.push(next_token);
              if (!delta.empty()) {
                json chunk_obj = {
                  {"id", completion_id},
                  {"object", "chat.completion.chunk"},
                  {"created", time(nullptr)},
                  {"model", active_adapter.empty() ? "gemma-4-e2b" : active_adapter},
                  {"choices", json::array({
                    {
                      {"index", 0},
                      {"delta", {{"content", delta}}},
                      {"finish_reason", nullptr}
                    }
                  })}
                };
                if (!req_user.empty()) chunk_obj["user"] = req_user;
                std::string sse = "data: " + chunk_obj.dump() + "\n\n";
                sink.write(sse.data(), sse.size());
              }
              count++;
              const float* nxt_logits = nullptr;
              {
                std::lock_guard<std::mutex> lock(g_engine_mutex);
                nxt_logits = engine_ptr->decode(uint32_t(slot_idx), next_token);
                next_token = sampler.sample(nxt_logits, tok_ptr->vocab_size(), step++, gen_history.data(), gen_history.size());
              }
            }

            std::string flushed = decoder.flush();
            if (!flushed.empty()) {
              json chunk_obj = {
                {"id", completion_id},
                {"object", "chat.completion.chunk"},
                {"created", time(nullptr)},
                {"model", active_adapter.empty() ? "gemma-4-e2b" : active_adapter},
                {"choices", json::array({
                  {
                    {"index", 0},
                    {"delta", {{"content", flushed}}},
                    {"finish_reason", nullptr}
                  }
                })}
              };
              if (!req_user.empty()) chunk_obj["user"] = req_user;
              std::string sse = "data: " + chunk_obj.dump() + "\n\n";
              sink.write(sse.data(), sse.size());
            }

            double decode_ms = ni::now_ms() - t_decode_start;
            double decode_tps = (decode_ms > 0 && count > 0) ? (double(count) / (decode_ms / 1000.0)) : 0.0;
            const char* finish_reason = (count >= req_max_tokens) ? "length" : "stop";

            // Final chunk with actual finish_reason
            json end_obj = {
              {"id", completion_id},
              {"object", "chat.completion.chunk"},
              {"created", time(nullptr)},
              {"model", active_adapter.empty() ? "gemma-4-e2b" : active_adapter},
              {"choices", json::array({
                {
                  {"index", 0},
                  {"delta", json::object()},
                  {"finish_reason", finish_reason}
                }
              })}
            };
            if (!req_user.empty()) end_obj["user"] = req_user;
            std::string sse_end = "data: " + end_obj.dump() + "\n\ndata: [DONE]\n\n";
            sink.write(sse_end.data(), sse_end.size());
            sink.done();
            slot_ptr->set_active(slot_idx, false);

            logi("[request] slot=%d user='%s' ctx=%zu (%zu prompt, %d gen) | prefill: %.1f tok/s (%.1f ms) | decode: %.1f tok/s (%.1f ms) | reason: %s",
                 slot_idx, req_user.empty() ? "anonymous" : req_user.c_str(),
                 prompt_ids.size() + count, prompt_ids.size(), count,
                 prefill_tps, prefill_ms, decode_tps, decode_ms, finish_reason);

            return true;
          });
      } else {
        // Non-streaming response
        ni::StreamDecoder decoder(tokenizer);
        std::string full_response;
        std::vector<int> gen_history;
        int step = 0;
        int count = 0;
        double t_decode_start = ni::now_ms();

        {
          std::lock_guard<std::mutex> lock(g_engine_mutex);
          int next_token = sampler.sample(logits, tokenizer.vocab_size(), step++, gen_history.data(), gen_history.size());

          while (count < req_max_tokens && !tokenizer.is_eos(next_token)) {
            gen_history.push_back(next_token);
            full_response += decoder.push(next_token);
            count++;
            const float* nxt_logits = engine.decode(uint32_t(slot_idx), next_token);
            next_token = sampler.sample(nxt_logits, tokenizer.vocab_size(), step++, gen_history.data(), gen_history.size());
          }
        }
        full_response += decoder.flush();
        slot_mgr.set_active(slot_idx, false);

        double decode_ms = ni::now_ms() - t_decode_start;
        double decode_tps = (decode_ms > 0 && count > 0) ? (double(count) / (decode_ms / 1000.0)) : 0.0;
        const char* finish_reason = (count >= req_max_tokens) ? "length" : "stop";

        logi("[request] slot=%d user='%s' ctx=%zu (%zu prompt, %d gen) | prefill: %.1f tok/s (%.1f ms) | decode: %.1f tok/s (%.1f ms) | reason: %s",
             slot_idx, req_user.empty() ? "anonymous" : req_user.c_str(),
             prompt_ids.size() + count, prompt_ids.size(), count,
             prefill_tps, prefill_ms, decode_tps, decode_ms, finish_reason);

        json resp_obj = {
          {"id", completion_id},
          {"object", "chat.completion"},
          {"created", time(nullptr)},
          {"model", active_adapter.empty() ? "gemma-4-e2b" : active_adapter},
          {"choices", json::array({
            {
              {"index", 0},
              {"message", {
                {"role", "assistant"},
                {"content", full_response}
              }},
              {"finish_reason", finish_reason}
            }
          })},
          {"usage", {
            {"prompt_tokens", prompt_ids.size()},
            {"completion_tokens", count},
            {"total_tokens", prompt_ids.size() + count}
          }}
        };
        if (!req_user.empty()) resp_obj["user"] = req_user;
        res.set_content(resp_obj.dump(), "application/json");
      }
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  std::thread watchdog_thread;
  if (idle_timeout_model_sec > 0 || idle_timeout_kv_sec > 0) {
    logi("gemma-amx: idle watchdog active (model timeout: %ds, kv timeout: %ds)",
         idle_timeout_model_sec, idle_timeout_kv_sec);
    watchdog_thread = std::thread([&, idle_timeout_model_sec, idle_timeout_kv_sec]() {
      while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!g_running.load()) break;

        double idle_sec = (ni::now_ms() - double(g_last_request_time_ms.load())) / 1000.0;

        // Model idle timeout: unload weights from RAM
        if (idle_timeout_model_sec > 0 && idle_sec >= double(idle_timeout_model_sec) && engine.is_model_loaded()) {
          std::lock_guard<std::mutex> lock(g_engine_mutex);
          if (engine.is_model_loaded()) {
            logi("gemma-amx: idle timeout reached (%.1fs >= %ds) - unloading model weights",
                 idle_sec, idle_timeout_model_sec);
            engine.unload_model();
          }
        }

        // KV cache idle timeout: unload slot arenas and snapshots
        if (idle_timeout_kv_sec > 0 && idle_sec >= double(idle_timeout_kv_sec) && engine.is_kv_loaded()) {
          std::lock_guard<std::mutex> lock(g_engine_mutex);
          if (engine.is_kv_loaded()) {
            logi("gemma-amx: idle timeout reached (%.1fs >= %ds) - unloading slot KV caches",
                 idle_sec, idle_timeout_kv_sec);
            engine.unload_kv();
          }
        }
      }
    });
  }

  logi("gemma-amx: listening on http://%s:%d (snapshot primed: %s, k: %s, v: %s, flash_attn: %s)",
       host.c_str(), port, engine.is_primed() ? "yes" : "no",
       engine.config().kv_type_k.c_str(), engine.config().kv_type_v.c_str(),
       engine.flash_attn_enabled() ? "yes" : "no");
  g_server->listen(host.c_str(), port);

  g_running = false;
  if (watchdog_thread.joinable()) {
    watchdog_thread.join();
  }

  logi("gemma-amx: shutting down");
  return 0;
}
