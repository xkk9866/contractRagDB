"""Precompute BGE embeddings for corpora. Saves fp16 .npy + ids json.

Usage: python scripts/embed_corpora.py hybridqa|asqa|crag
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.retrieval import embed_texts  # noqa: E402


def embed_jsonl(in_path, out_prefix, id_key, text_key, title_key=None, batch=512):
    ids, texts = [], []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d[id_key])
            t = d[text_key]
            if title_key and d.get(title_key):
                t = d[title_key] + "\n" + t
            texts.append(t[:2000])
    print(f"{len(texts)} texts to embed")
    t0 = time.time()
    emb = embed_texts(texts, batch_size=batch, show_progress=True)
    print(f"embedded in {time.time()-t0:.0f}s")
    np.save(out_prefix + ".npy", emb.astype(np.float16))
    with open(out_prefix + "_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print("saved", out_prefix)


def embed_crag_chunks():
    """CRAG chunks are per-query pools; store one npz per query for fast gather."""
    chunks_dir = os.path.join(ROOT, "data", "crag", "chunks")
    out_dir = os.path.join(ROOT, "data", "crag", "chunk_emb")
    os.makedirs(out_dir, exist_ok=True)
    qids = [f[:-5] for f in os.listdir(chunks_dir) if f.endswith(".json")]
    done = {f[:-4] for f in os.listdir(out_dir) if f.endswith(".npy")}
    todo = [q for q in qids if q not in done]
    print(f"{len(todo)} query pools to embed")
    for i, qid in enumerate(todo):
        chunks = json.load(open(os.path.join(chunks_dir, f"{qid}.json"), encoding="utf-8"))
        texts = [c["text"][:2000] for c in chunks]
        if not texts:
            np.save(os.path.join(out_dir, f"{qid}.npy"),
                    np.zeros((0, 768), dtype=np.float16))
            continue
        emb = embed_texts(texts, batch_size=256)
        np.save(os.path.join(out_dir, f"{qid}.npy"), emb.astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(todo)}", flush=True)
    print("CRAG chunk embeddings done")


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "hybridqa":
        embed_jsonl(os.path.join(ROOT, "data", "hybridqa", "passages.jsonl"),
                    os.path.join(ROOT, "data", "hybridqa", "passages_emb"),
                    "pid", "text")
    elif which == "asqa":
        embed_jsonl(os.path.join(ROOT, "data", "asqa", "corpus.jsonl"),
                    os.path.join(ROOT, "data", "asqa", "corpus_emb"),
                    "pid", "text", title_key="title")
    elif which == "qampari":
        embed_jsonl(os.path.join(ROOT, "data", "qampari", "corpus.jsonl"),
                    os.path.join(ROOT, "data", "qampari", "corpus_emb"),
                    "pid", "text", title_key="title")
    elif which == "crag":
        embed_crag_chunks()
    else:
        raise SystemExit("unknown corpus")
