import sqlite3

con = sqlite3.connect("data/llm_cache.sqlite")
print([r for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")])
for t in ["cache", "cache_gpu"]:
    try:
        print(t, con.execute(f"PRAGMA table_info({t})").fetchall())
    except Exception as e:
        print(t, "ERR", e)
try:
    for m in con.execute(
            "SELECT model, COUNT(*), AVG(prompt_tokens), AVG(completion_tokens) "
            "FROM cache GROUP BY model").fetchall():
        print(m)
except Exception as e:
    print("agg ERR", e)
