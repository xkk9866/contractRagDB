# ContractRAG artifact

This repository accompanies:

> ContractRAG: Risk-Certified Query Optimization for
> Retrieval-Augmented Data Systems

It contains the optimizer and runtime implementation, materialized
execution matrices, cached model-call records, experiment outputs, and
plotting scripts used in the paper.

## Repository map

- `contractrag/`: plan execution, policy construction, risk
  certification, baselines, planning, and anytime monitoring.
- `contractrag/tracks/`: HybridQA, CRAG, ASQA, and QAMPARI adapters.
- `scripts/`: dataset preparation, plan execution, experiments, system
  benchmarks, validation, and figure generation.
- `experiments/`: materialized train/calibration/test matrices, scores,
  and JSON outputs reported in the paper.
- `data/`: benchmark inputs and local indexes.
- `paper/VLDB2027/`: the PVLDB manuscript source.

## Environment

The cached-data workflow requires Python 3.10 or newer and the following
packages:

```text
numpy scipy scikit-learn optuna matplotlib requests beautifulsoup4
openai bm25s sentence-transformers torch huggingface-hub
```

The full re-execution workflow additionally requires the benchmark data,
model/API credentials, local model weights, and GPU capacity described in
the paper. Set provider credentials through environment variables used by
`contractrag/llm.py`; do not place credentials in the repository.

Run commands from the repository root so that the local `contractrag`
package is importable.

## Fast reproduction from materialized executions

The principal tables can be regenerated without issuing model calls.
Examples:

```bash
python scripts/experiment_main.py hybridqa --contract quality --tau 0.5 --alphas 0.34
python scripts/experiment_main.py crag --contract correct --tau 0.5 --alphas 0.65
python scripts/experiment_main.py asqa --contract citation --tau 50 --alphas 0.216
python scripts/experiment_main.py qampari --contract citation --tau 50 --alphas 0.71
```

Repeated-draw population-safety experiments:

```bash
python scripts/experiment_repeat.py hybridqa --contract quality --tau 0.5 --alpha 0.35 --repeats 1000
python scripts/experiment_repeat.py crag --contract correct --tau 0.5 --alpha 0.65 --repeats 1000
python scripts/experiment_repeat.py asqa --contract citation --tau 50 --alpha 0.25 --repeats 1000
python scripts/experiment_repeat.py qampari --contract citation --tau 50 --alpha 0.66 --repeats 1000
```

Finite-population validity and the exact hypergeometric recheck:

```bash
python scripts/experiment_finitepop.py --repeats 1000
```

Runtime and system experiments:

```bash
python scripts/experiment_drift.py --contract correct --alpha 0.66 --delta 0.1
python scripts/experiment_group.py --contract correct --alpha 0.65
python scripts/experiment_joint4.py
python scripts/experiment_planner.py
python scripts/bench_system.py
```

Regenerate the manuscript's quantitative plots after the JSON files are
present:

```bash
python scripts/make_figures.py all
```

Outputs are written to `experiments/`; figures are written to
`paper/figures/`. The manuscript records the exact parameter values used
for each reported table and figure.

## Full execution path

The full path is intended for users who want to replace the released
execution matrices or evaluate new models:

1. Prepare a benchmark with the matching `scripts/build_*.py` command.
2. Build dense/sparse resources with `scripts/embed_corpora.py` and the
   track adapter.
3. Execute all ladder rungs with `scripts/run_ladder.py`.
4. Produce loss and sufficiency records with `scripts/score_matrix.py`.
5. Run the cached-data experiments listed above.

API calls are cached in SQLite by model, messages, and decoding parameters.
The released matrices and JSON files are the authoritative inputs for the
paper's no-new-call reproduction path.

## Expected checks

- `experiments/finitepop_check.json` records the rejection-region
  dominance checks and repeated-draw exact-test results.
- `experiments/bench_system.json` records optimizer time, local-compute
  accounting, and shared-materialization results.
- `experiments/drift_crag_correct_a0.66.json` records the primary shift
  experiment.
- `experiments/repeat_*` files record method-level population failure
  probabilities and normalized cost.

Minor timing variation is expected across machines. Risk, selected-policy,
and cost values from the released execution matrices are deterministic for
a fixed random seed.

## License and external data

Add the intended software license before archival release. Benchmark
datasets and model weights retain their original licenses and should be
downloaded from their official sources when redistribution is not
permitted.

