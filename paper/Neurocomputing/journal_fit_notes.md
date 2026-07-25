# Neurocomputing positioning notes

Research checked: 25 July 2026.

## What the journal explicitly asks for

Neurocomputing seeks fundamental contributions to neural computation,
including neural architectures, learning methods, machine learning,
computational learning theory, optimization, and resource allocation:

- https://www.sciencedirect.com/journal/neurocomputing
- https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors

The guide specifies single-anonymized review, an abstract of no more
than 250 words, one to seven keywords, and three to five highlights of
at most 85 characters each.

## Recent RAG papers examined

| Paper | Methodological center | Breadth and analysis |
|---|---|---|
| AIR-RAG, 2026, doi:10.1016/j.neucom.2025.132272 | Adaptive feedback jointly refines ranking and retrieved content | Six public benchmarks; strong RAG baselines; retriever/LLM portability |
| RAG-LER, 2025, doi:10.1016/j.neucom.2025.131514 | LLM-supervised re-ranker with a confidence-weighted objective | Open-domain QA and fact checking; retriever/model transfer; latency |
| IRAGKR, 2025, doi:10.1016/j.neucom.2025.131282 | Dynamic retrieval gating, semantic query expansion, and knowledge compression | Component analysis of gating, expansion, and refinement |
| Cue RAG, 2025, doi:10.1016/j.neucom.2025.130235 | Cue memory and a new retrieval framework | Theoretical derivation and five public datasets |
| ICR, 2026, doi:10.1016/j.neucom.2025.132139 | Learned resolution of parametric/retrieved knowledge conflicts | Eight conflict types and public QA benchmarks |
| UA-RAG, 2026, doi:10.1016/j.neucom.2026.134357 | Retrieval triggered by complementary internal uncertainty signals | Five knowledge-intensive benchmarks and several LLM backbones |

## Recurring expression and argument pattern

1. Isolate one failure in the learning or inference mechanism.
2. Introduce a named model, objective, gate, memory, or update rule.
3. Explain why that mechanism changes the attainable accuracy-cost
   frontier.
4. Give either a theoretical argument or a mechanism-focused analysis.
5. Test on public benchmarks, compare with strong contemporaries, and
   ablate the proposed components.
6. Report efficiency after effectiveness and generality.

## Adaptation used in this submission

This paper does not introduce another retriever or re-ranker. Its
methodological contribution is a learning-and-certification layer that
can sit above them. Learned sufficiency scores choose whether a query
should stop or acquire stronger evidence; learned vouchers order
candidate policies. A distribution-free fixed-sequence procedure then
certifies the selected policy independently of score calibration.
An anytime-valid restart-mixture detector adapts the certified policy
under knowledge drift.

The Neurocomputing version therefore leads with progressive neural
inference, score-independent validity, cost consistency, and
change-time-independent detection delay. HybridQA, CRAG, ASQA, and
QAMPARI provide heterogeneous public evaluation, and the existing score,
policy-family, calibration-size, dense-grid, audit-rate, and drift
studies are presented as mechanism ablations.

## Honest scope boundary

RAG-LER, IRAGKR, AIR-RAG, and UA-RAG improve the underlying retrieval
or attainable answer-quality frontier. ContractRAG certifies which
policy on a supplied frontier may be deployed. The paper must not imply
head-to-head accuracy superiority over those methods without new
executions. They are complementary candidates that can be placed
inside the certified policy family.
