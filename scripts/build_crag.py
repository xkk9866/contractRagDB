"""Build Track-B (CRAG task 1&2) data.

Streams crag_task_1_and_2_dev_v4.jsonl.bz2; per query:
  - extracts text chunks from the 5 HTML search results (with page ages),
  - stores query metadata (domain, question_type, static_or_dynamic, query_time).

Outputs under data/crag/:
  queries.jsonl   {qid, query, answer, alt_ans, domain, question_type,
                   static_or_dynamic, popularity, query_time, split}
  chunks/{qid}.json  [{cid, text, page_name, page_url, age_hours, last_modified}]
"""
import bz2
import json
import os
import re
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "CRAG", "data", "crag_task_1_and_2_dev_v4.jsonl.bz2")
OUT = os.path.join(ROOT, "data", "crag")
os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)

CHUNK_TOKENS = 200  # approx words per chunk
MAX_CHUNKS_PER_PAGE = 40


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
    except Exception:
        return ""


def chunk_text(text: str):
    words = text.split()
    for i in range(0, len(words), CHUNK_TOKENS):
        yield " ".join(words[i:i + CHUNK_TOKENS])


def parse_time(s):
    if not s:
        return None
    s = s.strip()
    # CRAG query_time: "03/10/2024, 23:19:21 PT" (Pacific, ~UTC-8)
    if s.endswith(" PT"):
        try:
            dt = datetime.strptime(s[:-3], "%m/%d/%Y, %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc) + timedelta(hours=8)
        except Exception:
            pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def main():
    nq = 0
    qf = open(os.path.join(OUT, "queries.jsonl"), "w", encoding="utf-8")
    with bz2.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qid = ex["interaction_id"]
            qtime = parse_time(ex.get("query_time"))
            chunks = []
            for pi, page in enumerate(ex.get("search_results", [])):
                text = html_to_text(page.get("page_result", ""))
                if not text:
                    text = page.get("page_snippet", "") or ""
                age_h = None
                lm = parse_time(page.get("page_last_modified"))
                if lm is not None and qtime is not None:
                    age_h = max(0.0, (qtime - lm).total_seconds() / 3600.0)
                for ci, ch in enumerate(chunk_text(text)):
                    if ci >= MAX_CHUNKS_PER_PAGE:
                        break
                    chunks.append({
                        "cid": f"{pi}-{ci}", "text": ch,
                        "page_name": page.get("page_name", ""),
                        "page_url": page.get("page_url", ""),
                        "age_hours": age_h,
                        "last_modified": page.get("page_last_modified"),
                    })
                # always keep the snippet as a chunk (it is short and dense)
                snip = (page.get("page_snippet") or "").strip()
                if snip:
                    chunks.append({
                        "cid": f"{pi}-snippet", "text": snip,
                        "page_name": page.get("page_name", ""),
                        "page_url": page.get("page_url", ""),
                        "age_hours": age_h,
                        "last_modified": page.get("page_last_modified"),
                    })
            with open(os.path.join(OUT, "chunks", f"{qid}.json"), "w", encoding="utf-8") as cf:
                json.dump(chunks, cf, ensure_ascii=False)
            qf.write(json.dumps({
                "qid": qid, "query": ex["query"], "answer": ex["answer"],
                "alt_ans": ex.get("alt_ans", []), "domain": ex["domain"],
                "question_type": ex["question_type"],
                "static_or_dynamic": ex["static_or_dynamic"],
                "popularity": ex.get("popularity", ""),
                "query_time": ex.get("query_time"), "split": ex.get("split", 0),
            }, ensure_ascii=False) + "\n")
            nq += 1
            if nq % 100 == 0:
                print(f"{nq} queries processed", flush=True)
    qf.close()
    print(f"DONE {nq} queries")


if __name__ == "__main__":
    main()
