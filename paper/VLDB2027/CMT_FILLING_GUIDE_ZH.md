# PVLDB Volume 20（VLDB 2027）CMT 填写指南

## 先处理截止时间

August 2026 cycle 的摘要截止时间是 **2026-07-25 17:00 Pacific
Time**，即北京时间 **2026-07-26 08:00**。论文 PDF 截止时间是
**2026-08-01 17:00 Pacific Time**，即北京时间 **2026-08-02
08:00**。摘要现在应立即在 CMT 保存；正式截止时间以 CMT 页面显示为准。

## Title

```text
ContractRAG: Risk-Certified Query Optimization for Retrieval-Augmented Data Systems
```

## Abstract

```text
Retrieval-augmented data services execute queries over tables, text collections, vector indexes, and structured APIs. A query can therefore admit many physical plans with very different costs, but these plans are not semantically equivalent: cheaper retrieval, reranking, and generation choices may change the answer. Existing optimizers select plans from estimated quality and cost, so their selected plan has no finite-sample guarantee of meeting an application's quality, citation, freshness, or latency target. We present evidence contracts, a declarative interface that bounds service-level violation rates, and ContractRAG, a query optimizer that minimizes cost subject to such contracts. ContractRAG generates a plan lattice with cost-reducing rewrites, orders it using selectivity-aware risk vouchers, and constructs fixed, routed, and progressive-execution policies. A fixed-sequence statistical test then certifies the deployed policy without a multiplicity penalty from plan search. An anytime-valid monitor detects post-deployment drift and invokes recertification. Experiments on HybridQA, CRAG, ASQA, and QAMPARI use token-metered executions and exact finite-population risk over repeated calibration draws. ContractRAG is the only tested optimizer that stays within the prescribed delta = 0.1 failure budget at every contract level; its worst observed deployment-failure rate is 0.8%, whereas point-estimate optimizers reach 64%. At matched contracts, ContractRAG reduces cost by up to 20x relative to the strongest fixed plan and certifies plan spaces containing nearly 100,000 policies in seconds.
```

## Authors

按以下顺序添加，姓名和单位均使用拉丁字符：

1. Kang Yao — `yaokang@mail.ustc.edu.cn` — University of Science and
   Technology of China — China。建议设为 Primary Contact；若 CMT 当前账号
   使用其他邮箱，应把官方邮箱关联到同一 CMT 账号。
2. Shengjiang Zhang — `zhangshengjiang@mail.ustc.edu.cn` — University
   of Science and Technology of China — China。
3. Ronghao Pei — `peironghao@hikvision.com` — Hikvision Research
   Institute — China。
4. Weiwei Fu — `fuww@sibet.ac.cn` — Suzhou Institute of Biomedical
   Engineering and Technology, Chinese Academy of Sciences — China。
5. Jinjiang Cui — `cuijj@sibet.ac.cn` — Suzhou Institute of Biomedical
   Engineering and Technology, Chinese Academy of Sciences — China。

PDF 已注明 Kang Yao 与 Shengjiang Zhang 共同一作，Weiwei Fu 与
Jinjiang Cui 共同通讯。提交后作者集合不能更改，只能调整顺序，所以提交前
必须由五位作者共同确认。

## Subject Areas

Primary 选：

```text
Query processing and optimization
```

Secondary 建议选择以下 8 项，不必凑满 10 项：

```text
Runtime strategies and data access in ML systems
Compilation and optimization in ML systems
New data system infrastructures and tools for applied ML
Tuning, benchmarking, and performance measurement
Views, indexing, and search
Information retrieval
LLM-assisted data processing
Fuzzy, probabilistic, and approximate data
```

## File

上传：

```text
submission_files/final/VLDB2027_Manuscript.pdf
```

该 PDF 使用官方 PVLDB Volume 20 模板，共 10 页，正文及结论在第 9 页
结束，参考文献延伸至第 10 页；文件小于 10 MB。

## Mandatory confirmations

- Toronto Paper Matching System：`I agree`
- 12-month resubmission rule：只有在这项工作过去 12 个月没有被 PVLDB
  Research Track 拒稿时才能 `I agree`。若被拒过，不应提交本轮。
- Conflicts：所有作者先在各自 CMT 账号中完整登记 domain 和个人冲突，
  再选择 `I agree`。
- Integrity of review process：`I agree`
- Authors fixed after submission：五位作者确认后选 `I agree`
- Submission caps：确认每位作者本月不超过 2 篇、Volume 20 不超过 12
  篇，再选 `I agree`
- Paper category：`Regular Research Paper`

## Availability and reproducibility

选择：

```text
Yes
```

URL：

```text
https://github.com/xkk9866/contractRagDB
```

可填写说明：

```text
The public repository contains the ContractRAG implementation, materialized execution matrices, token-metered experiment outputs, the LLM-call cache, and scripts for reproducing the repeated-draw safety study, finite-population checks, cost comparisons, drift experiments, system benchmarks, tables, and figures. The repository README documents the software environment, cached-data workflow, output files, and full re-execution path.
```

重要：当前公开仓库没有根目录 README。提交前必须把本文件夹中的
`ARTIFACT_README.md` 整理为仓库根目录 `README.md` 并推送，随后用无痕
浏览器确认无需登录即可打开。最好再创建固定的 release/tag 或 Zenodo DOI，
避免主分支继续变化。

## Self-assessment of relevance

此项虽为 optional，但 RAG 属于相邻领域，建议填写：

```text
This paper addresses query processing and optimization for data services that access relational tables, text collections, vector indexes, and structured APIs. Section 2 defines a typed physical-plan model and a declarative evidence contract. Section 3 develops non-equivalent plan rewrites, selectivity-driven table/text access paths, shared materialization, adaptive physical policies, and a selection-valid optimizer. Sections 4 and 5 connect progressive execution, online monitoring, and recertification in a working system. Section 6 evaluates end-to-end execution cost, exact population-level contract validity, access-path selection, plan-space scale, optimizer overhead, and drift recovery. The contributions build directly on cost-based optimization, Cascades-style plan search, semantic operators, integrity constraints, and adaptive query execution.
```

## Contributing to the review board

Question 10 选择 `I agree`，但 Question 11 不能猜。

合格 reviewer 必须有 PhD，并且至少有 2 篇 SIGMOD、VLDB、ICDE、EDBT
或 CIDR 论文。现有材料不足以证明五位作者中有人满足这两个条件。请五位
作者核查 DBLP 后再填写：

- 若有人满足：填写其姓名、邮箱、PhD 和两篇符合条件的论文。
- 若无人满足：如实提名 best-qualified author。可使用以下模板，并将方括号
  中的事实补全：

```text
No author has two prior research papers in the listed database venues. We therefore nominate [NAME] ([EMAIL]) as the best-qualified author available. [NAME] holds a PhD and has research experience in [RELEVANT AREA]. All authors agree to this nomination.
```

不要在未核实学位和论文记录时直接提交这项。

## Related submissions

- 如果 KBS、ESWA、Neurocomputing 版本只是准备文件、从未投稿或已撤回，
  可填 `None.`
- 如果实质相同的稿件正在任何期刊/会议审稿，不能同时提交 PVLDB；必须先
  正式撤稿并确认撤稿完成。
- 如果存在不同但相关的同期稿件，则必须在正文参考文献中标注
  `under submission`，并在此处列出、说明与本稿的增量关系。

## AI/LLM usage disclosure

这是必填项，不能留空或隐瞒。根据本稿实际准备过程，建议如实填写：

```text
OpenAI Codex/ChatGPT was used for English-language editing, literature-search assistance, restructuring and drafting portions of the manuscript, and LaTeX and figure-caption preparation. The authors reviewed and verified the technical content, citations, code, data, and reported experimental results and remain fully responsible for the submission.
```

该说明填在 CMT，不需要另加到论文正文。

## Shadow PC and GDPR

- Shadow PC：建议选 `Yes, I consent...`，可获得额外反馈，且官方明确说明
  不影响正式审稿结果。若项目有保密要求，可选 No。
- GDPR：五位作者同意后选 `I agree`。

