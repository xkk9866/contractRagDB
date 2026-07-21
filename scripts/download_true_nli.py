"""Robust download of google/t5_xxl_true_nli_mixture weight shards via mirror.

Loops hf_hub_download per file (resumes .incomplete blobs) until all shards
are complete. Run with HF_ENDPOINT=https://hf-mirror.com.
"""
import time

from huggingface_hub import hf_hub_download

FILES = [f"pytorch_model-0000{i}-of-00005.bin" for i in range(1, 6)]

for f in FILES:
    for attempt in range(100):
        try:
            p = hf_hub_download("google/t5_xxl_true_nli_mixture", f,
                                etag_timeout=60)
            print("OK", f, p, flush=True)
            break
        except Exception as e:
            print(f"retry {attempt} {f}: {type(e).__name__} {str(e)[:90]}",
                  flush=True)
            time.sleep(10)
print("ALL_SHARDS_DONE")
