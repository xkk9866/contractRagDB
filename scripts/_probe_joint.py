import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData
from scripts.experiment_main import make_loss_fn, kg_costs_for

kgc = kg_costs_for("crag")
loss_fn = make_loss_fn("crag", "correct", 0.5)
td = TrackData("crag", "cal", loss_fn, kg_costs=kgc)
ld, qids, _, aux = td.build()
lat = aux["latency"]
cum_lat = np.cumsum(lat, axis=0)

# freshness / hallucination indicators need raw records
L, n = ld.L, ld.n
fresh = np.zeros((L, n))
hallu = np.zeros((L, n))
minage = np.zeros((L, n))
for i, qid in enumerate(td.qids):
    for j in range(L):
        rec, sc = td.records[qid][j]
        f = rec["features"]
        minage[j, i] = f.get("min_age_h", 0.0)
        fresh[j, i] = float(f.get("min_age_h", 0.0) > 72.0)
        hallu[j, i] = float(sc.get("label") == "incorrect")

print("per-rung correct-viol:", np.round(ld.losses.mean(axis=1), 3))
print("per-rung halluc     :", np.round(hallu.mean(axis=1), 3))
print("per-rung fresh-viol (min_age>72):", np.round(fresh.mean(axis=1), 3))
print("min_age quantiles rung0:", np.percentile(minage[0], [50, 75, 90, 95]))
for B in [5, 8, 10, 15, 20, 30]:
    print(f"lat>{B}s per rung:", np.round((cum_lat > B).mean(axis=1), 3))

# ASQA probe
loss_fn2 = make_loss_fn("asqa", "citation", 50.0)
td2 = TrackData("asqa", "cal", loss_fn2)
ld2, _, _, aux2 = td2.build()
lat2 = np.cumsum(aux2["latency"], axis=0)
L2, n2 = ld2.L, ld2.n
q_em = np.zeros((L2, n2))
c_rec = np.zeros((L2, n2))
c_prec = np.zeros((L2, n2))
for i, qid in enumerate(td2.qids):
    for j in range(L2):
        rec, sc = td2.records[qid][j]
        q_em[j, i] = float(sc.get("str_em", 0.0) < 30.0)
        c_rec[j, i] = float(sc.get("citation_rec", 0.0) < 50.0)
        c_prec[j, i] = float(sc.get("citation_prec", 0.0) < 50.0)
print("ASQA per-rung strEM<30:", np.round(q_em.mean(axis=1), 3))
print("ASQA per-rung citrec<50:", np.round(c_rec.mean(axis=1), 3))
print("ASQA per-rung citprec<50:", np.round(c_prec.mean(axis=1), 3))
for B in [5, 8, 10, 15]:
    print(f"ASQA lat>{B}s per rung:", np.round((lat2 > B).mean(axis=1), 3))
