import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData
from scripts.run_ladder import get_splits
from scripts.experiment_main import make_loss_fn, kg_costs_for

kgc = kg_costs_for("crag")
splits = get_splits("crag")
dyn = {q["qid"]: q["static_or_dynamic"] for s in splits.values() for q in s}

for contract in ["hallu", "correct"]:
    loss_fn = make_loss_fn("crag", contract, 0.5)
    td = TrackData("crag", "test", loss_fn, kg_costs=kgc)
    ld, qids, groups, aux = td.build(groups_map=dyn)
    pools = {
        "static": np.isin(groups, ["static", "slow-changing"]),
        "fast": groups == "fast-changing",
        "rt": groups == "real-time",
    }
    print("==", contract)
    for name, m in pools.items():
        print(f"  {name:>7s} n={m.sum():4d} per-rung "
              f"{np.round(ld.losses[:, m].mean(axis=1), 3)}")
