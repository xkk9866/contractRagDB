"""Precompute KG-API evidence for CRAG queries (APIRetrieve operator).

For each query: qwen-flash plans up to 3 mock-API calls (domain-filtered
catalog), we execute them against the local mock API server, and store the
formatted results + accounting under data/crag/kg_evidence/{qid}.json.

Requires the mock API server running:  uvicorn server:app --port 8000
(cwd = external/CRAG/mock_api)
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.llm import get_llm, Usage, call_cost_cny  # noqa: E402

DATA = os.path.join(ROOT, "data", "crag")
OUT = os.path.join(DATA, "kg_evidence")
os.makedirs(OUT, exist_ok=True)
SERVER = os.environ.get("CRAG_SERVER", "http://127.0.0.1:8000")

CATALOG = {
    "open": [
        "open/search_entity_by_name(query: str) -> entity names matching query",
        "open/get_entity(query: str) -> all KG facts about the exact entity name",
    ],
    "movie": [
        "movie/get_person_info(query: person name) -> acted_movies, directed_movies, oscar_awards, birthday",
        "movie/get_movie_info(query: movie title) -> cast, crew, budget, revenue, release_date, oscar_awards, rating",
        "movie/get_year_info(query: year like '2018') -> movies released that year, oscar awards that year",
    ],
    "finance": [
        "finance/get_ticker_by_name(query: company name) -> ticker symbol",
        "finance/get_price_history(query: ticker) -> last year of daily Open/High/Low/Close/Volume",
        "finance/get_detailed_price_history(query: ticker) -> minute-level price history",
        "finance/get_dividends_history(query: ticker) -> dividend events",
        "finance/get_market_capitalization(query: ticker) -> market cap",
        "finance/get_eps(query: ticker) -> earnings per share",
        "finance/get_pe_ratio(query: ticker) -> P/E ratio",
        "finance/get_info(query: ticker) -> company fundamentals dict",
    ],
    "music": [
        "music/search_artist_entity_by_name(query) -> artist names",
        "music/search_song_entity_by_name(query) -> song names",
        "music/get_billboard_rank_date(rank: int, date: 'YYYY-MM-DD') -> song+artist at rank on date",
        "music/get_billboard_attributes(date, attribute, song_name) -> attribute (rank_last_week/weeks_in_chart/top_position/rank) of song",
        "music/grammy_get_best_artist_by_year(query: year int) -> best new artist",
        "music/grammy_get_award_count_by_artist(query: artist) -> grammy count",
        "music/grammy_get_award_count_by_song(query: song) -> grammy count",
        "music/grammy_get_best_song_by_year(query: year int) -> song of the year",
        "music/grammy_get_award_date_by_artist(query: artist) -> years won",
        "music/grammy_get_best_album_by_year(query: year int) -> album of the year",
        "music/get_artist_birth_place(query: artist) -> birthplace",
        "music/get_artist_birth_date(query: artist) -> birth date",
        "music/get_members(query: band) -> members",
        "music/get_lifespan(query: artist) -> [birth, death]",
        "music/get_song_author(query: song) -> author",
        "music/get_song_release_country(query: song) -> country",
        "music/get_song_release_date(query: song) -> date",
        "music/get_artist_all_works(query: artist) -> all songs",
    ],
    "sports": [
        "sports/soccer/get_games_on_date(date: 'YYYY-MM-DD', team_name: str|null) -> soccer games/results on date",
        "sports/nba/get_games_on_date(date: 'YYYY-MM-DD', team_name: str|null) -> NBA games/results on date",
    ],
}

PLAN_PROMPT = """You are an API-call planner for a knowledge-graph API.
Current time: {qtime}
Question ({domain}): {query}

Available endpoints:
{catalog}

Plan up to 3 API calls that could retrieve facts to answer the question.
Chain lookups when needed (e.g., get_ticker_by_name before price queries is NOT
possible in one shot - instead just call get_ticker_by_name AND the price
endpoint with your best-guess ticker symbol).
Dates in questions like "yesterday/today/last week" are relative to current time.
Reply with ONLY a JSON array, each item: {{"endpoint": "<path>", "args": {{...}}}}.
Use arg name "query" for single-argument endpoints; billboard uses rank/date or
date/attribute/song_name; sports uses date/team_name. Reply [] if no endpoint helps."""


def call_api(endpoint, args, timeout=30):
    url = f"{SERVER}/{endpoint}"
    r = requests.post(url, json=args, headers={"accept": "application/json"},
                      timeout=timeout)
    return r.json()


def compact(obj, limit=1500):
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def summarize_price_history(result, query_time, limit=1500):
    """Price histories are huge; keep the 10 most recent trading days before
    query time."""
    try:
        hist = result["result"]
        if isinstance(hist, dict):
            items = sorted(hist.items())[-15:]
            return json.dumps(dict(items), ensure_ascii=False, default=str)[:limit]
    except Exception:
        pass
    return compact(result, limit)


def process_query(llm, q):
    out_path = os.path.join(OUT, f"{q['qid']}.json")
    if os.path.exists(out_path):
        return None
    domain = q["domain"] if q["domain"] in CATALOG else "open"
    catalog = "\n".join("- " + e for e in CATALOG[domain])
    usage = Usage()
    msgs = [{"role": "user", "content": PLAN_PROMPT.format(
        qtime=q.get("query_time"), domain=domain, query=q["query"], catalog=catalog)}]
    t0 = time.time()
    resp = llm.chat("qwen-flash", msgs, max_tokens=512, usage=usage)
    plan_cost = call_cost_cny("qwen-flash", resp["prompt_tokens"], resp["completion_tokens"])
    m = re.search(r"\[.*\]", resp["text"], re.DOTALL)
    calls = []
    if m:
        try:
            calls = json.loads(m.group(0))[:3]
        except Exception:
            calls = []
    evidence, call_log = [], []
    api_latency = 0.0
    for c in calls:
        try:
            ep = c.get("endpoint", "").strip("/")
            args = c.get("args", {})
            ta = time.time()
            result = call_api(ep, args)
            api_latency += time.time() - ta
            if "price_history" in ep:
                txt = summarize_price_history(result, q.get("query_time"))
            else:
                txt = compact(result)
            evidence.append(f"API {ep}({json.dumps(args, ensure_ascii=False)}) -> {txt}")
            call_log.append({"endpoint": ep, "args": args, "ok": True})
        except Exception as e:
            call_log.append({"endpoint": c.get("endpoint"), "args": c.get("args"),
                             "ok": False, "err": str(e)[:200]})
    rec = {"qid": q["qid"], "evidence": evidence, "calls": call_log,
           "plan_cost_cny": plan_cost, "plan_latency": resp["latency"],
           "api_latency": api_latency, "total_latency": time.time() - t0}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


def main():
    queries = []
    with open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"{len(queries)} queries")
    llm = get_llm(40)
    done = 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, _ in enumerate(ex.map(lambda q: process_query(llm, q), queries)):
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(queries)}", flush=True)
    print("KG_PRECOMPUTE_DONE")


if __name__ == "__main__":
    main()
