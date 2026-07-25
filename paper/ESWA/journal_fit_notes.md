# Expert Systems with Applications positioning notes

Research checked: 25 July 2026.

## What the journal explicitly asks for

Expert Systems with Applications focuses on the design, development,
testing, implementation, and management of expert and intelligent
systems. Its scope explicitly includes decision support, risk
assessment, information retrieval, intelligent databases, medicine,
engineering, finance, law, and other operational settings:

- https://www.sciencedirect.com/journal/expert-systems-with-applications
- https://www.sciencedirect.com/journal/expert-systems-with-applications/publish/guide-for-authors

The strongest fit is therefore a complete, tested decision system whose
method can be reused by practitioners. Pure component tuning is less
persuasive than an end-to-end workflow with an explicit operator
decision, a safe failure mode, and realistic operating costs.

## Recent RAG papers examined

| Paper | System contribution | Validation style |
|---|---|---|
| WADSeg, 2026, doi:10.1016/j.eswa.2025.129297 | A deployable attention-based knowledge-segmentation module | Multiple datasets, retrieval accuracy, scalability, plug-in use |
| SageRAG, 2026, doi:10.1016/j.eswa.2026.131160 | A research assistant with query rewriting and live scholarly retrieval | Grounding, answer quality, statistical tests, verifiable sources |
| SAGE, 2026, doi:10.1016/j.eswa.2026.131524 | A complete emotional-support system with retrieval, constrained decoding, re-ranking, and knowledge fusion | Automatic and human evaluation; operational strategy accuracy |
| Enterprise-specific QA for IT operations, 2026, doi:10.1016/j.eswa.2025.130961 | A full offline/online methodology for enterprise deployment | Domain-expert questions, multiple QA difficulties, real corpora |
| OntoLLM, 2026, doi:10.1016/j.eswa.2026.131505 | Ontology- and knowledge-graph-grounded industrial dialogue | Four industrial datasets and component ablations |

## Recurring expression and argument pattern

1. Open with a concrete decision or user problem.
2. Describe the complete system boundary and workflow.
3. Identify the mechanism that makes the system dependable in use.
4. Evaluate both task quality and operational consequences.
5. Include ablations, multiple data sources, and clear deployment
   guidance.

## Adaptation used in this submission

The ESWA version presents ContractRAG as a risk-aware expert system for
RAG operations. A domain expert declares acceptable failure rates; the
system compiles heterogeneous retrieval plans, learns progressive
policies, returns the least-cost certified policy, or explicitly
declines deployment. A runtime monitor closes the decision loop.

The abstract, introduction, discussion, and cover letter lead with that
operator-facing workflow. The statistical results remain the technical
basis, while the practical result is a reusable design for turning
service requirements into auditable deployment decisions. The
submission package uses a separate author title page and an anonymized
manuscript as requested.
