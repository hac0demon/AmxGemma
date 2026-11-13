import json
import os
import sys
import time
import urllib.request
import urllib.error

# Ensure loopback calls don't route through any proxy
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

BASE_URL = "http://127.0.0.1:8089"

def test_health():
    print("[TEST] GET /health... ", end="", flush=True)
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("status") == "ok"
        assert data.get("model") == "gemma-4-e2b"
        assert data.get("model_loaded") is True
        assert data.get("kv_loaded") is True
        assert data.get("flash_attn") is True
        assert data.get("kv_cache", {}).get("k_type") == "q8_0"
        assert data.get("kv_cache", {}).get("v_type") == "q4_0"
        assert "yarn" in data
        yarn_info = data["yarn"]
        assert "enabled" in yarn_info
        print(f"PASSED (k_type: {data.get('kv_cache',{}).get('k_type')}, v_type: {data.get('kv_cache',{}).get('v_type')}, flash_attn: {data.get('flash_attn')}, primed: {data.get('primed')}, context: {data.get('context_max')}, yarn_enabled: {yarn_info.get('enabled')}, yarn_scale: {yarn_info.get('scale')})")

def test_models():
    print("[TEST] GET /v1/models... ", end="", flush=True)
    req = urllib.request.Request(f"{BASE_URL}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("object") == "list"
        models = [m["id"] for m in data.get("data", [])]
        assert "gemma-4-e2b" in models
        print(f"PASSED (models: {models})")

def test_prime():
    print("[TEST] POST /v1/prime... ", end="", flush=True)
    payload = json.dumps({
        "system_prompt": "You are a specialized log compaction assistant. Summarize errors concisely."
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/prime",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("status") == "primed"
        assert data.get("tokens", 0) > 0
        print(f"PASSED (primed {data.get('tokens')} tokens)")

def test_chat_completions_non_streaming():
    print("[TEST] POST /v1/chat/completions (non-streaming with user parameter)... ", end="", flush=True)
    payload = json.dumps({
        "model": "gemma-4-e2b",
        "messages": [
            {"role": "system", "content": "You are a specialized log compaction assistant. Summarize errors concisely."},
            {"role": "user", "content": "2026-11-12 08:15:00 ERROR [db] Connection refused to 10.0.0.1:5432"}
        ],
        "max_tokens": 25,
        "temperature": 0.0,
        "user": "end-user-test-42"
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("object") == "chat.completion"
        assert data.get("user") == "end-user-test-42", f"Expected user 'end-user-test-42', got {data.get('user')}"
        choices = data.get("choices", [])
        assert len(choices) > 0
        msg = choices[0]["message"]["content"]
        assert len(msg.strip()) > 0
        print(f"PASSED\n  User echo: {data.get('user')}\n  Output: {msg.strip()}")

def test_chat_completions_streaming():
    print("[TEST] POST /v1/chat/completions (streaming SSE with user parameter)... ", end="", flush=True)
    payload = json.dumps({
        "model": "gemma-4-e2b",
        "messages": [
            {"role": "user", "content": "Ping"}
        ],
        "max_tokens": 10,
        "stream": True,
        "user": "stream-user-77"
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200
        chunks = []
        user_seen = False
        saw_done = False
        for line in resp:
            line_str = line.decode().strip()
            if not line_str or not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                saw_done = True
                break
            chunk_data = json.loads(data_str)
            if chunk_data.get("user") == "stream-user-77":
                user_seen = True
            delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                chunks.append(delta)

        assert user_seen, "User parameter was not present in streamed chunks"
        assert saw_done, "Stream did not terminate with [DONE]"
        full_text = "".join(chunks).strip()
        print(f"PASSED\n  User confirmed in SSE chunks\n  Streamed output: {full_text}")

def test_slots_and_adapters():
    print("[TEST] GET /v1/slots & Adapters API... ", end="", flush=True)
    req = urllib.request.Request(f"{BASE_URL}/v1/slots")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        slots = json.loads(resp.read().decode())
        assert len(slots) >= 4, f"Expected >= 4 slots, got {len(slots)}"
    
    # Test attaching adapter to user "reducer"
    attach_payload = json.dumps({
        "user": "reducer",
        "adapter": "logs_reducer"
    }).encode()
    req_attach = urllib.request.Request(
        f"{BASE_URL}/v1/slots/attach",
        data=attach_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_attach, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("status") == "ok"
        assert data.get("target") == "reducer"
    print(f"PASSED ({len(slots)} slots available, attached 'logs_reducer' to user 'reducer')")

def test_simultaneous_multi_user():
    print("[TEST] Simultaneous access by 4 distinct users (including 'reducer')... ", end="", flush=True)
    import concurrent.futures

    users = ["user_alpha", "user_beta", "user_gamma", "reducer"]
    prompts = [
        "What is 1+1? Answer with single digit.",
        "What is 2+2? Answer with single digit.",
        "What is 3+3? Answer with single digit.",
        "ERROR 2026-11-12 [log_reducer] Service crashed on signal 9"
    ]

    def send_req(user_id, prompt):
        payload = json.dumps({
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 12,
            "temperature": 0.0,
            "user": user_id
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return user_id, data

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_req, u, p) for u, p in zip(users, prompts)]
        for f in concurrent.futures.as_completed(futures):
            u, data = f.result()
            results[u] = data

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    for u in users:
        assert u in results
        data = results[u]
        assert data.get("user") == u, f"Expected echoed user {u}, got {data.get('user')}"
        choices = data.get("choices", [])
        assert len(choices) > 0
        content = choices[0]["message"]["content"]
        assert len(content.strip()) > 0

    print("PASSED")
    for u in users:
        out = results[u]["choices"][0]["message"]["content"].strip().replace("\n", " ")
        print(f"  [{u}] -> {out[:60]}")

def test_unload_and_reload():
    print("[TEST] POST /v1/unload (freeing KV) & auto-reload on demand... ", end="", flush=True)
    # 1. Unload KV cache
    payload = json.dumps({"target": "kv"}).encode()
    req = urllib.request.Request(f"{BASE_URL}/v1/unload", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("kv_loaded") is False
        assert data.get("model_loaded") is True

    # 2. Check health reflects unloaded KV
    req_h = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req_h, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        assert data.get("kv_loaded") is False

    # 3. Send new chat completion request - should auto-rehydrate KV without error
    req_chat = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps({
            "model": "gemma-4-e2b",
            "messages": [{"role": "user", "content": "1+1="}],
            "max_tokens": 5,
            "temperature": 0.0,
            "user": "reload-test-user"
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_chat, timeout=30) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data.get("user") == "reload-test-user"

    # 4. Check health confirms KV is reloaded
    with urllib.request.urlopen(req_h, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        assert data.get("kv_loaded") is True

    print("PASSED (KV unloaded, verified via /health, auto-reloaded on demand)")

if __name__ == "__main__":
    time.sleep(1) # Wait for server ready
    test_health()
    test_models()
    test_slots_and_adapters()
    test_prime()
    test_chat_completions_non_streaming()
    test_chat_completions_streaming()
    test_simultaneous_multi_user()
    test_unload_and_reload()
    print("=== All HTTP Server Tests Passed Successfully ===")
