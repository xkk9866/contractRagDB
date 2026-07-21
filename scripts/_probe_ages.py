import json
import os

import numpy as np

EXP = "experiments"


def probe(tag, split):
    per_rung = {}
    with open(os.path.join(EXP, f"{tag}_{split}_matrix.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("evidence_meta", {})
            srcs = meta.get("srcs", [])
            ages = meta.get("ages", [])
            dated = [a for s, a in zip(srcs, ages)
                     if s == "web" and a is not None]
            stale = float(any(a > 72.0 for a in dated)) if dated else 0.0
            per_rung.setdefault(r["rung"], []).append(stale)
    for j in sorted(per_rung):
        v = np.array(per_rung[j])
        print(f"{tag} {split} rung {j}: n={len(v)} stale-any>72h rate={v.mean():.3f}")


probe("crag", "cal")
probe("crag", "test")
