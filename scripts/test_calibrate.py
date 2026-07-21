"""Synthetic validity check for the certification machinery.

Simulates a 3-rung ladder where higher rungs are more accurate and more
expensive, with informative sufficiency scores. Verifies:
1. certified policies keep test-set risk <= alpha (with prob >= 1-delta over
   repeated calibration draws);
2. certified policies are much cheaper than always running the final rung;
3. the e-process alarm fires under injected drift and stays quiet without it.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.calibrate import (LadderData, certify_ladder, policy_risk_cost,
                                   EProcessMonitor, hb_p_value)

rng = np.random.default_rng(0)


def make_world(n, seed):
    rng = np.random.default_rng(seed)
    # latent difficulty in [0,1]
    diff = rng.uniform(0, 1, n)
    # rung error rates decrease with rung, increase with difficulty
    err = np.stack([
        np.clip(0.05 + 0.9 * diff, 0, 1),      # cheap rung: bad on hard queries
        np.clip(0.02 + 0.45 * diff, 0, 1),     # mid rung
        np.clip(0.01 + 0.12 * diff, 0, 1),     # strong rung
    ])
    losses = (rng.uniform(size=err.shape) < err).astype(float)
    # sufficiency scores: higher when query easy; noisy
    scores = np.stack([
        1 - diff + 0.25 * rng.normal(size=n),
        1 - diff + 0.20 * rng.normal(size=n),
        np.full(n, np.inf),  # final rung always stops
    ])[:2]  # thresholds only needed for first L-1 rungs; keep L x n shapes below
    scores = np.vstack([scores, np.full((1, n), np.inf)])
    costs = np.stack([np.full(n, 1.0), np.full(n, 4.0), np.full(n, 20.0)])
    return LadderData(losses, scores, costs)


alpha, delta = 0.2, 0.1
violations = 0
trials = 200
costs_sel, costs_full = [], []
for t in range(trials):
    cal = make_world(400, seed=1000 + t)
    test = make_world(2000, seed=5000 + t)
    pol = certify_ladder(cal, alpha, delta)
    r_test, c_test = policy_risk_cost(test, pol.thresholds)
    violations += (r_test > alpha)
    costs_sel.append(c_test)
    _, c_full = policy_risk_cost(test, np.full(cal.L - 1, np.inf))
    costs_full.append(c_full)

print(f"test-risk > alpha in {violations}/{trials} trials "
      f"(should be <~ delta+noise = {delta})")
print(f"mean cost: certified={np.mean(costs_sel):.2f} vs always-final={np.mean(costs_full):.2f} "
      f"({np.mean(costs_sel)/np.mean(costs_full):.1%})")

# e-process: no drift -> quiet; drift -> alarm
mon = EProcessMonitor(alpha=0.2, delta=0.05)
quiet = rng.uniform(size=3000) < 0.15  # true rate below alpha
alarms_quiet = sum(mon.update(float(l)) for l in quiet)
mon2 = EProcessMonitor(alpha=0.2, delta=0.05)
drift = np.concatenate([rng.uniform(size=500) < 0.15, rng.uniform(size=1500) < 0.45])
fired = [mon2.update(float(l)) for l in drift]
print(f"e-process: quiet alarms={alarms_quiet} (want 0), "
      f"drift alarm at t={mon2.alarm_at} (drift starts at 500)")

# HB p-value sanity
print("hb_p(0.10, n=400, alpha=0.2) =", hb_p_value(0.10, 400, 0.2))
print("hb_p(0.25, n=400, alpha=0.2) =", hb_p_value(0.25, 400, 0.2))
