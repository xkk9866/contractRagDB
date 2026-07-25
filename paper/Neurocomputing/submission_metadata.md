# Submission metadata

## Title

Distribution-free certification of learned progressive policies for
retrieval-augmented generation under knowledge drift

## Abstract

Dynamic retrieval-augmented generation (RAG) learns when a language
model should retrieve, rerank, or escalate to a stronger generator.
These policies can improve the accuracy–cost trade-off, but their
scores do not bound population risk and may become unreliable after
knowledge drift. We introduce ContractRAG, a learning-and-certification
method for progressive RAG inference. Candidate policies use learned
sufficiency scores over retrieval and generation signals to stop at the
cheapest adequate rung. Risk vouchers summarize the observed effect of
plan rewrites and order a large policy family. Soundness comes from a
separate distribution-free fixed-sequence test: with probability at
least 1−δ, every loss in the selected evidence contract meets its
target, regardless of candidate count or score quality. We also prove a
cost-consistency result and give a restart-mixture e-process whose
drift-detection delay is independent of the change time. A geometric
schedule preserves validity through repeated recertification.
Experiments on HybridQA, CRAG, ASQA, and QAMPARI cover four model
families, two serving backends, and ablations of scoring, policy
families, calibration size, candidate density, audit rate, and drift.
Across 1000 calibration draws, selected policies violate their
contracts in at most 0.8% of draws under a 10% error budget, whereas
empirical and Bayesian selection reach 64%. The policies reduce model
cost by up to 20× (10–18× with audits). The method is orthogonal to the
underlying retriever or reranker and provides a general certification
layer for learned adaptive RAG.

## Keywords

Retrieval-augmented generation; Adaptive computation; Distribution-free
risk control; Uncertainty estimation; Drift detection; Large language
models
