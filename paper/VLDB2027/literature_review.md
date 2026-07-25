# PVLDB positioning review

## Official requirements used

- Venue: PVLDB Volume 20 / VLDB 2027 Research Track.
- Category: Regular Research Paper.
- Limit: 12 pages for all content, appendices, and acknowledgements;
  references alone may extend beyond the limit.
- Review model: single blind; author names and affiliations must appear on
  the first page.
- Template: the current official VLDB template based on `acmart` v2.19 and
  `pvldb.sty`. The mandatory PVLDB reference, license, and artifact blocks
  are retained.
- Scope test: core data-management problem, a substantive connection to
  database literature, and evaluation in a data-management context.
- Artifact: public repository with reusable code/data and meaningful
  instructions.

Official pages:

- <https://www.vldb.org/2027/submission-guidelines.html>
- <https://www.vldb.org/2027/formatting-guidelines.html>
- <https://www.vldb.org/2027/call-for-research-track.html>
- <https://github.com/vldbproceedings/VLDB-Template>

## Representative accepted papers studied

The PDFs below were downloaded from the official PVLDB site into
`literature/`.

| Paper | PVLDB record | Relevance to this submission |
|---|---|---|
| Abacus: A Cost-Based Optimizer for Semantic Operator Systems | 19(5), 1060–1073, 2026 | Poses semantic AI processing as constrained physical-plan optimization and evaluates quality, cost, and latency jointly. |
| KEN: An Execution Engine for Unstructured Database Systems | 19(5), 902–916, 2026 | Treats model cascades as physical implementations inside an execution engine and grounds adaptivity in DBMS runtime design. |
| DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing | 18(9), 3035–3048, 2025 | Uses a declarative interface, logical rewrites, plan evaluation, and an optimizer for document workloads. |
| LEGO-GraphRAG | 18(10), 3269–3283, 2025 | Decomposes GraphRAG into a modular design space and supports claims with large-scale, multidimensional evaluation. |
| Semantic Integrity Constraints | 18(11), 4073–4080, 2025 | Frames reliability of AI-augmented data processing as a declarative database constraint and connects it to planning/runtime enforcement. |
| LOTUS: Semantic Operators and Their Optimization | 18(11), 4171–4184, 2025 | Defines semantic operators, multiple physical algorithms, optimization, and statistical accuracy guarantees. |
| QUEST: Query Optimization in Unstructured Document Analysis | 18(11), 4560–4573, 2025 | Anchors LLM document processing in access paths, operator ordering, indexes, and end-to-end cost reduction. |

## Common characteristics of the accepted papers

1. **The database problem appears before the model.** They begin with an
   execution, data access, declarative interface, or query optimization
   problem rather than a generic LLM capability.
2. **They name a database abstraction.** Examples include semantic
   operators, integrity constraints, logical rewrites, physical
   implementations, cascades, indexes, and cost models.
3. **The system has a complete path from interface to runtime.** The paper
   does not stop at an algorithm: it explains planning, execution, and a
   working prototype.
4. **Evaluation follows the system claim.** Results include end-to-end
   quality/cost/latency, optimizer baselines, ablations, plan-space or
   runtime overhead, and reusable artifacts.
5. **Database literature is used substantively.** Classical query
   optimization, access paths, cascades, adaptive execution, and
   constraints shape the method instead of appearing only as citations.

## Resulting positioning for ContractRAG

The VLDB version is written as a data-system paper:

- evidence contracts are a declarative service-level constraint;
- heterogeneous RAG pipelines are physical plans over typed operators;
- non-equivalent rewrites form a plan lattice;
- selectivity estimates and risk vouchers guide enumeration;
- fixed, routed, and progressive policies are alternative physical
  execution strategies;
- a fixed-sequence test is the feasibility layer that remains valid after
  plan search;
- an e-process monitor and recertification loop provide adaptive runtime
  control under drift.

The title, abstract, introduction, related work, and evaluation were
rewritten around this database contribution. Claims centered only on model
performance or engineering cost were reduced or tied to a query optimizer,
execution engine, or data-management experiment.

