# AGENTS.md: AI Agent Guide for Gemma-AMX

This document serves as the operational guide and architecture manual for **autonomous AI agents, LLM toolchains, and AI coding assistants** (e.g., Antigravity, Claude, Copilot, Cursor) working with or developing the **Gemma-AMX** codebase.

---

## 1. Mission & System Overview

`gemma-amx` is a dedicated, zero-GPU inference sidecar engineered for **Google Gemma 4 E2B**. It is designed specifically to run alongside heavy GPU-bound language models (such as Qwen-32B or Llama-3-70B running on Metal or CUDA) without competing for VRAM, GPU command queues, or unified memory wired pages.

### Primary Use Cases for Autonomous Agents
1. **Log Compaction & Error Extraction**: Processing massive continuous log streams (up to 256K tokens) and extracting root-cause stack traces into compact summaries.
2. **Context Reduction for Long-Context Models**: Condensing multi-turn agent history or large code repositories into dense summaries before forwarding to 1M+ context models.
3. **Always-On Auxiliary Assistant**: Running background classification, query decomposition, and guardrail validation on CPU without blocking primary GPU inference.

---

## 2. Agent Interaction Protocol (Using Gemma-AMX as a Client)

When an agent interacts with `gemma-amx` over HTTP (default `http://127.0.0.1:8081`):

### 2.1 Always Specify the `user` Parameter
`gemma-amx` supports **simultaneous multi-user slots** ($N \ge 4$). Always pass a unique string in the `user` field of the OpenAI request. This binds the request to an isolated slot and retains sequence position and KV cache across turns:

```json
{
  "model": "gemma-4-e2b",
  "user": "agent-log-compactor",
  "messages": [
    {"role": "user", "content": "Analyze the following log payload..."}
  ],
  "max_tokens": 256,
  "temperature": 0.0,
  "repetition_penalty": 1.15
}
```

### 2.2 Exploit KV State Snapshot Priming
If an agent will submit multiple requests sharing an identical static system prompt (e.g. a 2,000-token prompt defining log reduction rules):
1. **Prime once**: Call `POST /v1/prime` with `{"system_prompt": "<your static prompt>"}`.
2. **Subsequent requests**: When sending requests with matching system prompts (or omitting system prompt entirely), `gemma-amx` bypasses prefill and restores the primed KV arena via `std::memcpy` in **115 microseconds** (`0.11 ms`), giving near-zero Time-To-First-Token (TTFT).

### 2.3 Direct Reducer Workflows to Attached Adapters
To run a specialized task (e.g. a fine-tuned log reducer adapter):
- Either pass `"adapter": "<adapter_id>"` or `"model": "<adapter_id>"` in the request body.
- Or attach the adapter permanently to the agent's user identifier:
  ```bash
  POST /v1/slots/attach
  {"user": "agent-log-compactor", "adapter": "logs_reducer"}
  ```

### 2.4 Managing Memory & Idle Timeouts
- Inspect current memory state via `GET /health` (`model_loaded`, `kv_loaded`, `flash_attn`).
- If an agent completes an intense workload and wants to release RAM immediately:
  ```bash
  POST /v1/unload
  {"target": "all"}  # Options: "model", "kv", "all"
  ```
- Any subsequent request will transparently auto-reload model weights and KV buffers in ~300 ms.

---

## 3. Codebase Architecture Map (For Coding Agents)

Coding agents modifying or debugging `GemmaAmx` should navigate the repository as follows:

```text
/Users/ivash/moe/GemmaAmx/
├── CMakeLists.txt              # Build configuration: C++20, -framework Accelerate
├── README.md                   # Human-facing documentation & benchmarks
├── AGENTS.md                   # Agent architecture manual (this file)
├── tests/
│   ├── test_amx.cpp            # C++ unit tests (AMX matmul, quantization, FA, YaRN)
│   └── test_http.py            # End-to-end Python HTTP integration test suite
└── src/
    ├── engine/
    │   ├── gemma4.h            # Config, quantization blocks, KV structs, Engine API
    │   ├── gemma4.cpp          # Forward pass, Flash Attention kernel, RoPE with YaRN
    │   └── sampler.h           # Top-K / Top-P / Greedy sampler
    ├── server/
    │   └── main.cpp            # HTTP server, SlotManager, Watchdog thread, OpenAI endpoints
    ├── tokenizer/
    │   ├── tokenizer.h         # SPM-BPE tokenizer interface
    │   └── tokenizer.cpp       # BPE encoder/decoder, GGUF metadata parser
    ├── gguf/
    │   ├── gguf.h              # Zero-copy mmap GGUF file parser
    │   ├── gguf.cpp            # Tensor table & metadata loading
    │   ├── quant.h             # Block quantization definitions
    │   └── quant.cpp           # CPU dequantizers & NEON dot products
    └── util/
        ├── log.h               # Logging macros (logi, loge, logd)
        └── bf16.h              # Fast FP32 <-> BF16 and FP16 conversions
```

### Core Classes & Data Structures

| Type | Location | Purpose |
| :--- | :--- | :--- |
| `Gemma4Engine` | `src/engine/gemma4.h` | Core inference engine: chunked prefill, decode, KV management, YaRN, LoRA. |
| `block_q8_0` | `src/engine/gemma4.h` | Packed 34-byte Key cache block: FP16 scale $d$ + 32 `int8_t` values. |
| `block_q4_0` | `src/engine/gemma4.h` | Packed 18-byte Value cache block: FP16 scale $d$ + 16 bytes (32 nibbles). |
| `UserSlot` | `src/engine/gemma4.h` | Isolated user session holding independent contiguous KV arena & adapter binding. |
| `KVSnapshot` | `src/engine/gemma4.h` | Snapshot buffer storing primed KV arena bytes for instant `memcpy` restoration. |
| `LoraAdapter` | `src/engine/gemma4.h` | Loaded low-rank matrices ($A$ and $B$) with scaling factor $\alpha / r$. |
| `YarnConfig` | `src/engine/gemma4.h` | RoPE YaRN configuration (wavelength ramp dims, attention temperature scale). |
| `SlotManager` | `src/server/main.cpp` | Thread-safe slot allocator and LRU router for multi-user access. |

---

## 4. Strict Engineering Rules for Agents

When implementing new features, refactoring, or optimizing code in this repository, you **MUST** adhere to the following rules:

### Rule 1: Strictly Zero Metal & Zero GPU Linkage
- **Never** include `<Metal/Metal.h>`, `<MetalKit/MetalKit.h>`, or `<MetalPerformanceShaders/MetalPerformanceShaders.h>`.
- **Never** link `-framework Metal` in `CMakeLists.txt`.
- Verification command: `otool -L build/gemma-amx` must list only `Accelerate.framework`, `CoreFoundation.framework`, and standard C/C++ runtime libraries.

### Rule 2: Packed Quantization Layouts (`#pragma pack(push, 1)`)
- Quantized block structures (`block_q8_0`, `block_q4_0`) must be packed with `#pragma pack(push, 1)` to match GGUF binary formats and prevent compiler structure padding.
- `sizeof(block_q8_0)` must equal **34 bytes**.
- `sizeof(block_q4_0)` must equal **18 bytes**.

### Rule 3: Memory Budgeting & Attention Matrix Materialization
- **Do not** allocate $O(N)$ attention score buffers during the forward pass.
- Attention **must** be computed via `flash_attn_head()` using online softmax tiling ($B_c = 32$) with stack-allocated accumulators ($< 2\text{ KB}$).
- Intermediate logits must be trimmed during prefill (`num_logits_to_keep = 1`). Never project the 262,144-element vocabulary on non-final prefill tokens.

### Rule 4: Apple Accelerate AMX Conformance
- Multi-token prefill ($T > 1$) **must** use `cblas_sgemm` to target the AMX matrix coprocessor tiles.
- Single-token decode ($T = 1$) **must** use `vDSP_dotpr` or vector operations to avoid matrix-setup overhead on small vectors.

### Rule 5: Pure C++20 & Thread Safety
- The engine forward pass is single-threaded per execution step protected by `g_engine_mutex`.
- Slot state modifications in `SlotManager` must be protected by its internal `std::mutex`.

---

## 5. Agent Development Recipes

### Recipe 1: Extending YaRN Beyond 256K Context (e.g. 512K)
If an agent needs to extend context further:
1. Pass `--ctx 512k` to the CLI (or configure `cfg.max_context = 524288`).
2. YaRN will auto-calculate:
   $$\text{scale} = \frac{524288}{131072} = 4.0$$
   $$\text{eff\_mscale} = 1.0 \cdot (1.0 + 0.1 \ln(4.0)) \approx 1.1386$$
3. Full Attention layers will automatically scale their RoPE rotation frequencies, while SWA layers remain fixed to 512 tokens.

### Recipe 2: Adding a New Quantization Type for KV Cache
To implement a new format (e.g. `q8_0` for both K and V):
1. Define the block layout in `src/engine/gemma4.h`.
2. Add the quantization function `quantize_<type>()` in `src/engine/gemma4.cpp`.
3. In `Gemma4Engine::load_model`: update `lw.k_token_bytes` and `lw.v_token_bytes`.
4. In `flash_attn_head`: update tile dequantization loop for the new format.
5. Add round-trip accuracy verification in `tests/test_amx.cpp`.

### Recipe 3: Fine-Tuning or Merging LoRA Adapters for Log Compaction
To adapt attention head weights for extended 256K context or specific log formats:
1. Train a LoRA adapter targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, and `mlp` projections.
2. Save as standard GGUF or Safetensors.
3. Pass `--adapter <name>:<path>` at launch, or load at runtime via `POST /v1/adapters/load`.
4. Bind to a dedicated user slot with `POST /v1/slots/attach`.

### Recipe 4: Verifying Regressions
Before completing any task, run both test suites:
```bash
# 1. Compile and run C++ unit tests:
cmake --build build -j8 && ./build/test_amx

# 2. Run end-to-end Python HTTP integration test:
python3 tests/test_http.py
```
Both test suites must exit with return code `0`.
