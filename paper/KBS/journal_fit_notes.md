# Knowledge-Based Systems positioning notes

Research checked: 25 July 2026.

## What the journal explicitly asks for

Knowledge-Based Systems states that it publishes original, innovative,
and creative work on knowledge-based and AI systems. Its scope stresses
knowledge representation and engineering, intelligent models and
methods, support for human prediction and decision-making, and a balance
between theory and practice:

- https://www.sciencedirect.com/journal/knowledge-based-systems
- https://www.sciencedirect.com/journal/knowledge-based-systems/about/news/knowledge-based-systems-kbs-outstanding-paper-award

The 2025 outstanding-paper criteria also name novelty, technical
quality, significance, impact, and academic metrics. This makes a
"better accuracy at lower cost" story too narrow unless the paper first
establishes a new knowledge representation or reasoning mechanism.

## Recent RAG papers examined

| Paper | What is placed first | Evidence used to support the claim |
|---|---|---|
| Flow-RAG, 2026, doi:10.1016/j.knosys.2026.116400 | A knowledge-graph reasoning representation (time-state trellis) and gated flow propagation | WebQSP and ComplexWebQuestions; generalization and latency |
| Toward trustworthy engineering information extraction using RAG, 2026, doi:10.1016/j.knosys.2026.116418 | A unified trustworthy knowledge-acquisition framework with traceable sources | Retrieval and extraction accuracy, hard negatives, source grounding |
| Retrieval augmented generation using engineering design knowledge, 2024, doi:10.1016/j.knosys.2024.112410 | Explicit engineering knowledge represented as subject-relation-object triples | A large relation dataset and a multi-million-fact knowledge base |
| Automated data synthesis and RAG for legal LLMs, 2026, doi:10.1016/j.knosys.2026.116267 | A knowledge-generation and evaluation process for a high-stakes domain | LawBench, model judging, and ELO-style comparison |

An open author version of the engineering-design paper is stored in
`literature/engineering_design_rag_author_version.pdf`.

## Recurring expression and argument pattern

1. Start from a knowledge problem, not from a generic LLM limitation.
2. Name the representation or inference object early.
3. Explain how the object changes reasoning, accountability, or
   decision support.
4. State formal or algorithmic properties before reporting efficiency.
5. Use heterogeneous or public tasks to show that the method is not a
   one-off application.

## Adaptation used in this submission

The KBS version treats an evidence contract as a machine-checkable
knowledge representation. Each constraint connects an evidence
property, an admissible violation rate, and a certification confidence.
Risk vouchers form compositional knowledge about plan rewrites.
Fixed-sequence inference turns that representation into a certified
deployment decision, and the e-process updates the decision when the
knowledge environment drifts. Cost reduction is reported as a
consequence of certified reasoning, not as the main theoretical claim.

The paper does not claim that vouchers alone prove reliability. The
distribution-free test is the source of soundness; learned scores and
vouchers only order the search. That separation is central to the KBS
positioning.
