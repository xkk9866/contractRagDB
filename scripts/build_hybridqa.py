"""Build Track-A (HybridQA) data: tables + passage corpus + query files.

Streams WikiTables-WithLinks tar.gz directly (some member names are invalid
NTFS paths, so we never materialize them on disk).

Outputs under data/hybridqa/:
  tables.jsonl    {table_id, url, title, section_title, header, rows(tokenized),
                   row_links[list per row of passage url ids]}
  passages.jsonl  {pid(url), text}
  queries_{train,dev}.jsonl  {qid, question, table_id, answer}
"""
import io
import json
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAR = os.path.join(ROOT, "external", "wikitables.tar.gz")
OUT = os.path.join(ROOT, "data", "hybridqa")
os.makedirs(OUT, exist_ok=True)


def main():
    n_tables = n_passages = 0
    tables_f = open(os.path.join(OUT, "tables.jsonl"), "w", encoding="utf-8")
    passages_f = open(os.path.join(OUT, "passages.jsonl"), "w", encoding="utf-8")

    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            _, kind, fname = parts[0], parts[1], "/".join(parts[2:])
            if kind == "tables_tok" and fname.endswith(".json"):
                data = json.load(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"))
                table_id = fname[:-5]
                header = [h[0] for h in data.get("header", [])]
                rows, row_links = [], []
                for row in data.get("data", []):
                    cells, links = [], []
                    for cell in row:
                        cells.append(cell[0])
                        links.extend(cell[1])
                    rows.append(cells)
                    row_links.append(links)
                tables_f.write(json.dumps({
                    "table_id": table_id, "url": data.get("url", ""),
                    "title": data.get("title", ""),
                    "section_title": data.get("section_title", ""),
                    "intro": data.get("intro", ""), "header": header,
                    "rows": rows, "row_links": row_links,
                }, ensure_ascii=False) + "\n")
                n_tables += 1
            elif kind == "request_tok" and fname.endswith(".json"):
                data = json.load(io.TextIOWrapper(tf.extractfile(m), encoding="utf-8"))
                # each request_tok file maps {url: passage_text}
                for url, text in data.items():
                    passages_f.write(json.dumps(
                        {"pid": url, "text": text}, ensure_ascii=False) + "\n")
                    n_passages += 1
            if (n_tables + n_passages) % 50000 == 0:
                print(f"tables={n_tables} passages={n_passages}", flush=True)

    tables_f.close()
    passages_f.close()
    print(f"DONE tables={n_tables} passages={n_passages}")

    for split in ["train", "dev"]:
        src = os.path.join(ROOT, "external", "HybridQA", "released_data", f"{split}.json")
        data = json.load(open(src, encoding="utf-8"))
        with open(os.path.join(OUT, f"queries_{split}.jsonl"), "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps({
                    "qid": ex["question_id"], "question": ex["question"],
                    "table_id": ex["table_id"], "answer": ex.get("answer-text", ""),
                }, ensure_ascii=False) + "\n")
        print(split, len(data))


if __name__ == "__main__":
    main()
