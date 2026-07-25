# Submission metadata

## Title

ContractRAG: A risk-certified expert system for deploying
retrieval-augmented generation services

## Abstract

Operators of retrieval-augmented generation (RAG) services must choose
among many retrieval, reranking, and generation configurations. Average
development-set scores do not show whether the chosen service will
respect operational limits on wrong, unsupported, stale, or late
answers. We present ContractRAG, a risk-aware expert system that turns
such limits into auditable deployment decisions. A domain expert
specifies an evidence contract containing bounded losses, admissible
violation rates, and a confidence budget. The system compiles
heterogeneous knowledge-access plans, learns progressive policies that
acquire stronger evidence only when needed, and orders them with
composable risk vouchers. A distribution-free fixed-sequence test then
returns the least-cost policy in the certified prefix or reports that
the contract is infeasible. After deployment, an anytime-valid monitor
detects drift, identifies a fresh recalibration window, and restarts
the decision process under a lifetime error budget. We test the
complete workflow on HybridQA, CRAG, ASQA, and QAMPARI with four large
language model families and two serving backends. Across 1000
calibration draws, the deployed policies violate their contracts in at
most 0.8% of draws under a 10% error budget; empirical and Bayesian
tuning reach 64%. The system reduces model cost by up to 20×, or
10–18× after audit cost, and explicitly rejects impossible contracts.
ContractRAG therefore provides a reusable decision-support architecture
for reliable and economical RAG operations.

## Keywords

Expert systems; Retrieval-augmented generation; Decision support; Risk
assessment; Information retrieval; Runtime monitoring
