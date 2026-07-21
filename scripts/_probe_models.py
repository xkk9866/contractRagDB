import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from contractrag.llm import get_llm, Usage

MODELS = ["deepseek-v4-flash", "deepseek-v3.2", "deepseek-v4-pro",
          "glm-4.7", "glm-5", "glm-5.2", "kimi-k2.5",
          "ollama/gemma3:1b", "ollama/gemma3:4b", "ollama/gemma3:12b",
          "ollama/llama3.2:1b"]

l = get_llm()
for m in MODELS:
    try:
        u = Usage()
        r = l.chat(m, [{"role": "user", "content": "Reply with only: OK"}],
                   max_tokens=8, usage=u)
        print("%-22s OK  pt=%d ct=%d gpu_s=%.2f cost=%.6f  text=%r"
              % (m, r["prompt_tokens"], r["completion_tokens"],
                 r.get("gpu_s", -1), u.cost_cny, r["text"][:20]))
    except Exception as e:
        print("%-22s FAIL %s" % (m, str(e)[:100]))
