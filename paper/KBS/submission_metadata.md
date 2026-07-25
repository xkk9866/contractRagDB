# Submission metadata

## Title

Evidence contracts for certified knowledge acquisition in
retrieval-augmented generation systems

## Abstract

Reliable retrieval-augmented generation (RAG) requires more than
relevant passages: a knowledge service must represent what counts as
adequate evidence, reason over alternative acquisition plans, and
justify deployment from finite calibration data. We introduce evidence
contracts, a machine-checkable representation that links answer
quality, citation completeness, evidence freshness, and latency to
admissible violation rates and a confidence budget. ContractRAG
realizes these contracts over heterogeneous knowledge sources. It
represents plan transformations as a rewrite lattice, attaches
composable risk vouchers to rewrite edges, and learns progressive
policies that acquire stronger evidence only when needed. Vouchers
guide the search but do not determine validity. A distribution-free
fixed-sequence procedure certifies the selected policy with probability
at least 1−δ, regardless of candidate count or score quality; a
cost-consistency result characterizes when certification recovers the
least-cost feasible policy. After deployment, a restart-mixture
e-process detects knowledge drift, localizes the change, and triggers
recertification while preserving a lifetime error budget. Fully metered
experiments on HybridQA, CRAG, ASQA, and QAMPARI verify the guarantee
against exact finite-population risk. Across 1000 calibration draws,
the selected policies violate their contracts in at most 0.8% of draws
under a 10% error budget, whereas empirical and Bayesian tuning reach
64%. The certified policies also reduce model cost by up to 20×
(10–18× after audit cost). The results show how explicit knowledge
obligations can support auditable, adaptive decisions in RAG systems.

## Keywords

Knowledge representation; Retrieval-augmented generation; Reasoning
certification; Distribution-free inference; Knowledge drift; Decision
support
