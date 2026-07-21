"""Multi-provider LLM client with disk cache, parallel execution, cost/latency accounting.

Providers:
  - DashScope OpenAI-compatible endpoint: Qwen, DeepSeek, GLM, Kimi, ... families
    (token-metered at published CNY list prices).
  - Local Ollama (models prefixed "ollama/"): open-weight families (Gemma, Llama, ...)
    metered in GPU-seconds and priced at an RTX-4090 cloud rental rate, so local
    and hosted plans are comparable under one total-cost-of-ownership currency.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from openai import OpenAI

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-37755bfe12dc494fb6a4e2bf0a578873")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_PREFIX = "ollama/"
OLLAMA_MAX_PARALLEL = int(os.environ.get("OLLAMA_MAX_PARALLEL", "4"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

# RTX-4090 on-demand cloud rental, CNY per GPU-second (~2.0 CNY/hour); used to
# price local open-weight models by their measured per-request compute time.
GPU_RATE_CNY_S = 2.0 / 3600.0

# CNY per 1M tokens (input, output). DashScope list prices, 2026-07.
PRICES = {
    # Qwen family (Alibaba)
    "qwen-flash": (0.15, 1.5),
    "qwen-turbo": (0.3, 0.6),
    "qwen-plus": (0.8, 2.0),
    "qwen-max": (2.4, 9.6),
    "qwen2.5-7b-instruct": (0.5, 1.0),
    "qwen2.5-14b-instruct": (1.0, 3.0),
    "qwen2.5-32b-instruct": (2.0, 6.0),
    "qwen2.5-72b-instruct": (4.0, 12.0),
    # DeepSeek family (hosted on DashScope; list prices match DeepSeek's own API)
    "deepseek-v4-flash": (1.0, 2.0),
    "deepseek-v4-pro": (12.0, 24.0),
    "deepseek-v3.2": (2.0, 3.0),
    # GLM family (Zhipu, hosted on DashScope)
    "glm-4.7": (3.0, 14.0),
    "glm-5": (4.0, 18.0),
    "glm-5.2": (8.0, 28.0),
    # Kimi family (Moonshot, hosted on DashScope)
    "kimi-k2.5": (4.0, 21.0),
}

# Hybrid-reasoning models default to thinking mode on DashScope; we disable it
# (answers here are short spans / cited sentences, chains-of-thought only cost).
_NO_THINKING = ("glm-", "deepseek-v4", "deepseek-v3", "kimi-")


def _extra_body(model: str):
    if any(model.startswith(p) for p in _NO_THINKING):
        return {"enable_thinking": False}
    return None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(_ROOT, "data", "llm_cache.sqlite")


class LLMCache:
    """Thread-safe SQLite cache for LLM responses."""

    def __init__(self, path=CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._local = threading.local()
        self._path = path
        con = self._conn()
        con.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, "
            "prompt_tokens INT, completion_tokens INT, latency REAL, ts REAL)"
        )
        # GPU compute seconds for locally served (ollama) models
        con.execute("CREATE TABLE IF NOT EXISTS cache_gpu (key TEXT PRIMARY KEY, gpu_s REAL)")
        con.execute("PRAGMA journal_mode=WAL")
        con.commit()

    def _conn(self):
        if not hasattr(self._local, "con"):
            self._local.con = sqlite3.connect(self._path, timeout=60)
        return self._local.con

    def get(self, key):
        row = self._conn().execute(
            "SELECT response, prompt_tokens, completion_tokens, latency FROM cache WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        out = {"text": row[0], "prompt_tokens": row[1], "completion_tokens": row[2],
               "latency": row[3], "cached": True}
        g = self._conn().execute("SELECT gpu_s FROM cache_gpu WHERE key=?", (key,)).fetchone()
        if g is not None:
            out["gpu_s"] = g[0]
        return out

    def put(self, key, model, resp):
        for _ in range(5):
            try:
                self._conn().execute(
                    "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?)",
                    (key, model, resp["text"], resp["prompt_tokens"],
                     resp["completion_tokens"], resp["latency"], time.time()),
                )
                if "gpu_s" in resp:
                    self._conn().execute(
                        "INSERT OR REPLACE INTO cache_gpu VALUES (?,?)",
                        (key, resp["gpu_s"]))
                self._conn().commit()
                return
            except sqlite3.OperationalError:
                time.sleep(0.2)


@dataclass
class Usage:
    """Accumulates cost across calls (thread-safe)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_cny: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model, pt, ct, gpu_s=None):
        with self._lock:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.calls += 1
            self.cost_cny += call_cost_cny(model, pt, ct, gpu_s)


def call_cost_cny(model, pt, ct, gpu_s=None):
    """Cost of one call. Hosted models: token-metered at list prices.
    Local (ollama/) models: measured GPU compute seconds x rental rate."""
    if model.startswith(OLLAMA_PREFIX):
        return (gpu_s or 0.0) * GPU_RATE_CNY_S
    pin, pout = PRICES.get(model, (1.0, 2.0))
    return (pt * pin + ct * pout) / 1e6


class LLM:
    def __init__(self, max_workers=40):
        self.client = OpenAI(api_key=API_KEY, base_url=DASHSCOPE_BASE, timeout=120, max_retries=2)
        self.cache = LLMCache()
        self.max_workers = max_workers
        self._ollama_sem = threading.Semaphore(OLLAMA_MAX_PARALLEL)

    @staticmethod
    def _key(model, messages, temperature, max_tokens, seed):
        blob = json.dumps({"m": model, "msg": messages, "t": temperature,
                           "mt": max_tokens, "s": seed}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ------------------------------------------------------------------
    def _chat_ollama(self, model, messages, temperature, max_tokens, seed):
        """Native ollama /api/chat call. Returns response dict with exact
        per-request GPU compute time (prompt_eval + eval durations)."""
        import requests
        name = model[len(OLLAMA_PREFIX):]
        body = {
            "model": name, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "seed": seed, "num_ctx": OLLAMA_NUM_CTX},
        }
        with self._ollama_sem:
            t0 = time.time()
            r = requests.post(f"{OLLAMA_BASE}/api/chat", json=body, timeout=600)
            r.raise_for_status()
            d = r.json()
            dt = time.time() - t0
        gpu_s = (d.get("prompt_eval_duration", 0) + d.get("eval_duration", 0)) / 1e9
        return {
            "text": (d.get("message") or {}).get("content", "") or "",
            "prompt_tokens": int(d.get("prompt_eval_count", 0)),
            "completion_tokens": int(d.get("eval_count", 0)),
            "latency": dt, "gpu_s": gpu_s, "cached": False,
        }

    def chat(self, model, messages, temperature=0.0, max_tokens=1024, seed=0,
             use_cache=True, usage: Usage | None = None):
        """Single chat call. Returns dict with text/tokens/latency/cached."""
        key = self._key(model, messages, temperature, max_tokens, seed)
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                # cached calls still count into cost accounting (reproducible economics)
                if usage is not None:
                    usage.add(model, hit["prompt_tokens"], hit["completion_tokens"],
                              hit.get("gpu_s"))
                return hit
        last_err = None
        for attempt in range(10):
            try:
                if model.startswith(OLLAMA_PREFIX):
                    out = self._chat_ollama(model, messages, temperature, max_tokens, seed)
                else:
                    t0 = time.time()
                    r = self.client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens, seed=seed,
                        extra_body=_extra_body(model),
                    )
                    dt = time.time() - t0
                    out = {
                        "text": r.choices[0].message.content or "",
                        "prompt_tokens": r.usage.prompt_tokens,
                        "completion_tokens": r.usage.completion_tokens,
                        "latency": dt, "cached": False,
                    }
                self.cache.put(key, model, out)
                if usage is not None:
                    usage.add(model, out["prompt_tokens"], out["completion_tokens"],
                              out.get("gpu_s"))
                return out
            except Exception as e:  # rate limits, transient network
                last_err = e
                msg = str(e)
                if "data_inspection_failed" in msg or "inappropriate" in msg:
                    # provider content filter: treat as an empty (failed) answer
                    out = {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
                           "latency": 0.0, "cached": False, "filtered": True}
                    self.cache.put(key, model, out)
                    return out
                if "429" in msg or "quota" in msg or "limit" in msg.lower():
                    # rate limited: exponential backoff with jitter, up to ~90s
                    import random
                    time.sleep(min(90, 5 * 2 ** min(attempt, 4)) * (0.5 + random.random()))
                else:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def chat_batch(self, model, list_of_messages, temperature=0.0, max_tokens=1024,
                   seed=0, usage: Usage | None = None, desc=None):
        """Parallel batch. Preserves order. Returns list of response dicts."""
        workers = self.max_workers
        if model.startswith(OLLAMA_PREFIX):
            workers = min(workers, OLLAMA_MAX_PARALLEL)
        results = [None] * len(list_of_messages)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(self.chat, model, msgs, temperature, max_tokens, seed, True, usage): i
                for i, msgs in enumerate(list_of_messages)
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results


_default_llm = None


def get_llm(max_workers=40) -> LLM:
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM(max_workers=max_workers)
    return _default_llm
