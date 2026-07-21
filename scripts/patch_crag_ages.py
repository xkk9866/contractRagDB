"""Patch age_hours into existing data/crag/chunks/*.json.

The original build failed to parse CRAG's query_time format, leaving all
ages None. Re-streams the bz2 dump, recomputes per-page ages with the
fixed parser, and rewrites only the age fields (chunk texts untouched).
"""
import bz2
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_crag import parse_time  # noqa: E402

SRC = os.path.join(ROOT, "external", "CRAG", "data",
                   "crag_task_1_and_2_dev_v4.jsonl.bz2")
CHUNKS = os.path.join(ROOT, "data", "crag", "chunks")


def main():
    n = n_aged = 0
    with bz2.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid = ex["interaction_id"]
            path = os.path.join(CHUNKS, f"{qid}.json")
            if not os.path.exists(path):
                continue
            qtime = parse_time(ex.get("query_time"))
            ages = {}
            for pi, page in enumerate(ex.get("search_results", [])):
                lm = parse_time(page.get("page_last_modified"))
                if lm is not None and qtime is not None:
                    ages[pi] = max(0.0, (qtime - lm).total_seconds() / 3600.0)
            chunks = json.load(open(path, encoding="utf-8"))
            changed = False
            for c in chunks:
                pi = int(c["cid"].split("-")[0])
                new_age = ages.get(pi)
                if c.get("age_hours") != new_age:
                    c["age_hours"] = new_age
                    changed = True
                    n_aged += 1
            if changed:
                with open(path, "w", encoding="utf-8") as cf:
                    json.dump(chunks, cf, ensure_ascii=False)
            n += 1
            if n % 300 == 0:
                print(f"{n} queries patched ({n_aged} chunks aged)", flush=True)
    print(f"DONE {n} queries, {n_aged} chunks got ages")


if __name__ == "__main__":
    main()
