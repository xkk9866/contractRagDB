# 一、先把方向收窄：不要做泛化的“多目标 RAG 调参器”

原来的题目：

> **QOptRAG: Quality-Constrained Adaptive Query Optimization for Hybrid RAG**

范围仍然偏宽。更建议改成：

> **ContractRAG: Evidence-Contract-Aware Query Optimization for Hybrid and Dynamic RAG**

中文可以表述为：

> **面向混合动态数据的证据契约驱动 RAG 查询优化器**

核心不是“自动选择 top-k、模型和 reranker”，而是：

> 给定用户对答案质量、证据完整性、新鲜度、延迟和成本的要求，自动生成并动态调整跨 SQL、向量、文本、知识图谱和 API 的 RAG 执行计划。

Abacus 已经能够在语义算子系统中，根据少量验证样本、先验和 LLM Judge，搜索满足质量、成本、延迟约束的物理计划，并使用类似 Cascades 的计划搜索机制。因此，新论文不能再把“质量—成本—延迟约束优化”本身作为主要创新。

RAG-Stack 也已经提出 RAG-IR、RAG 成本模型和计划搜索三层框架。虽然其公开描述更接近研究蓝图，但这意味着“定义一个 RAG IR，再训练一个 cost model”也不足以构成强创新。([arXiv][1])

你的论文应重点突出三个现有工作没有完整解决的问题：

1. **跨异构数据源的 RAG 计划等价与近似等价规则；**
2. **把 freshness 和 citation completeness 变成真正的执行契约；**
3. **数据更新、API 延迟和索引状态变化后的在线重新优化。**

不建议第一篇同时加入 source authorization。权限安全可以成为后续独立论文，否则系统范围会过大。

---

# 二、论文核心问题形式化

## 2.1 输入

给定用户问题 (q)、异构数据源集合：

[
\mathcal D=
{D_{\text{sql}},D_{\text{bm25}},D_{\text{vec}},
D_{\text{kg}},D_{\text{api}}}
]

以及一个查询契约：

[
C(q)=
\langle
\tau_q,\tau_c,\Delta_f,B_l,B_$, \delta
\rangle
]

其中：

* (\tau_q)：最低答案质量；
* (\tau_c)：最低 citation completeness；
* (\Delta_f)：证据最大允许陈旧时间；
* (B_l)：延迟预算，例如 p95 不超过 3 秒；
* (B_$)：调用和推理成本预算；
* (\delta)：约束违反概率上限。

## 2.2 优化目标

选择物理执行计划 (P)：

[
\min_P \mathbb E[\operatorname{Cost}(P,q)]
]

满足：

[
\operatorname{LCB}_{1-\delta_q}
\big(Q(P,q)\big)
\ge \tau_q
]

[
\operatorname{LCB}_{1-\delta_c}
\big(Citation(P,q)\big)
\ge \tau_c
]

[
\max_{e\in E(P,q)}
\operatorname{Age}(e)
\le \Delta_f
]

[
Q_{0.95}\big(
\operatorname{Latency}(P,q)
\big)
\le B_l
]

这里的 LCB 是质量或引用完整性的统计下置信界。这样优化器不是根据平均质量选择计划，而是要求：

> 该计划以至少 (1-\delta) 的置信度满足用户契约。

这比普通加权目标更合理。普通目标：

[
Q-\lambda_1 Cost-\lambda_2 Latency
]

可能选择一个成本低但不满足最低质量要求的计划，而契约模型明确区分：

* 必须满足的硬约束；
* 可以优化的目标。

---

# 三、RAG 声明式查询语言

建议把 RAG 表示为下列逻辑算子。

| 算子                   | 功能                 |
| -------------------- | ------------------ |
| `StructuredRetrieve` | SQL、表格、属性过滤与聚合     |
| `SparseRetrieve`     | BM25 文本检索          |
| `DenseRetrieve`      | 向量近邻检索             |
| `GraphExpand`        | 实体或关系多跳扩展          |
| `APIRetrieve`        | Web/KG/实时 API 查询   |
| `EvidenceJoin`       | 跨表、文本、图和 API 的证据连接 |
| `Normalize`          | 实体、日期、单位和模式对齐      |
| `Rerank`             | 文档或证据重排序           |
| `Verify`             | 判断证据是否支持候选 claim   |
| `Generate`           | 生成最终答案             |
| `CiteCheck`          | 原子 claim 与引用证据匹配   |
| `ContractCheck`      | 检查质量、新鲜度、引用和延迟契约   |

例如一个查询可以表示为：

[
Generate(
Verify(
Rerank(
EvidenceJoin(
StructuredRetrieve(q),
DenseRetrieve(q),
GraphExpand(q)
))))
]

## 物理实现选择

同一个逻辑算子可以有多个物理实现。

例如 `DenseRetrieve` 可以选择：

* 精确向量检索；
* HNSW；
* 不同 `efSearch`；
* metadata pre-filter；
* ANN 后过滤；
* BM25 与 dense 融合；
* 小 embedding 模型或大 embedding 模型。

`Rerank` 可以选择：

* 不使用 reranker；
* cross-encoder；
* 小型 LLM；
* 大型 LLM；
* 分层 rerank；
* 对部分候选 rerank。

`Verify` 可以选择：

* NLI 模型；
* 小型 LLM verifier；
* 大型 LLM verifier；
* 多 verifier 投票。

这会产生大量物理计划，但论文不能只研究这些参数，而必须研究**算子顺序和数据源执行顺序**。

---

# 四、最重要的创新：带误差预算的近似计划等价

传统数据库中，可以使用严格的等价变换：

[
(A\Join B)\Join C
\equiv
A\Join(B\Join C)
]

但 RAG 中的大量变换并不严格等价。

例如：

[
Filter(DenseTopK(D,q))
]

与：

[
DenseTopK(Filter(D),q)
]

通常并不严格等价，因为 ANN、top-k 截断和过滤顺序都会改变候选集合。

因此建议定义：

[
P_1
\simeq_{\epsilon,\delta}
P_2
]

含义是：

[
\Pr[
d(E(P_1),E(P_2))\le \epsilon
]
\ge 1-\delta
]

其中 (E(P)) 是执行计划产生的证据集合。

每条近似重写规则都带有：

* 质量损失上界 (\epsilon_i)；
* 失败概率 (\delta_i)；
* 延迟或成本收益。

多个规则组合后：

[
\sum_i \epsilon_i\le \epsilon_C
]

[
\sum_i \delta_i\le \delta_C
]

优化器只能选择不超过查询契约误差预算的转换。

## 可以提出的具体重写规则

### 规则一：结构化过滤下推

[
DenseRetrieve(
SQLFilter(D,\phi),q,k
)
]

替代：

[
SQLFilter(
DenseRetrieve(D,q,k'),\phi
)
]

优化器根据过滤选择率、ANN 参数和召回风险，选择 pre-filter 或 post-filter。

### 规则二：SQL-first 与 Vector-first

对于包含结构化条件的问题：

[
Join(
SQLRetrieve(q_s),
VectorRetrieve(q_t)
)
]

可以采用：

* SQL 先缩小实体候选，再检索文本；
* 先向量检索，再读取结构化属性；
* SQL 与向量并行执行，再连接。

当 SQL 条件选择率很高时，SQL-first 通常成本更低；当自然语言条件难以结构化时，Vector-first 可能具有更高召回率。

### 规则三：Graph-expand 与 Retrieve 顺序交换

计划一：

[
GraphExpand(
VectorRetrieve(q)
)
]

计划二：

[
VectorRetrieve(
GraphExpandEntities(q)
)
]

前者先找到语义相关实体，后者先从问题抽取实体并扩展关系。二者对多跳问题的成本和召回差异可能非常大。

### 规则四：Verify 下推

原始计划：

[
Verify(
Rerank(
Merge(E_1,E_2,E_3)
))
]

近似计划：

[
Rerank(
Merge(
Verify(E_1),
Verify(E_2),
Verify(E_3)
))
]

提前删除低可信证据可以减少 rerank 和生成成本，但 verifier 本身也可能产生假阴性，因此需要消耗质量误差预算。

### 规则五：延迟生成与渐进式检索

先执行轻量计划：

[
P_0=
BM25\rightarrow SmallGenerator
]

如果 `ContractCheck` 判断证据不足，再升级：

[
P_1=
BM25+Dense
\rightarrow Rerank
\rightarrow Generator
]

如果引用覆盖仍不足，再执行：

[
P_2=
P_1+Graph/API
\rightarrow Verify
\rightarrow LargeGenerator
]

这种机制不是静态选择一个 pipeline，而是**运行时渐进执行**。

---

# 五、系统架构

## 5.1 Query Analyzer

从问题中提取：

* 是否包含日期或“目前、截至、最新”等时间表达；
* 是否存在明确结构化约束；
* 需要几跳推理；
* 是否涉及比较、排序、聚合；
* 是否需要多个独立来源；
* 是否适合 SQL、文本、图或 API；
* 预估答案长度和 claim 数量。

输出查询特征：

[
x(q)=
[
type,
hops,
entities,
temporal,
aggregation,
source\ requirements
]
]

## 5.2 Evidence Catalog

不仅保存数据库传统统计信息，还要保存：

* 各数据源检索召回率；
* 索引更新时间；
* 文档版本时间；
* API 更新频率；
* API 延迟分布；
* 每种物理算子的成本；
* verifier 精确率和召回率；
* 不同查询类别上的 citation coverage；
* LLM token 成本和延迟分布。

## 5.3 Contract-Cascades Optimizer

在 Cascades memo 中，每个候选计划不再只有一个 cost，而是保存：

[
\langle
Cost,
Latency_{p95},
Quality_{LCB},
Citation_{LCB},
Freshness,
\epsilon,
\delta
\rangle
]

两个计划 (P_1,P_2) 满足以下条件时，可以剪枝 (P_2)：

[
Cost(P_1)\le Cost(P_2)
]

[
Latency(P_1)\le Latency(P_2)
]

[
Quality_{LCB}(P_1)\ge Quality_{LCB}(P_2)
]

[
Citation_{LCB}(P_1)\ge Citation_{LCB}(P_2)
]

并且 (P_1) 的 freshness 不差于 (P_2)。

这实际上是一个多维 Pareto frontier，但你的关键区别是：

> Pareto 状态中包含证据新鲜度、引用完整性和近似重写误差预算。

## 5.4 Risk-Calibrated Estimator

不能只用一个 LLM Judge 预测质量，否则会产生评价闭环。

建议分别估计：

1. **Evidence sufficiency**：检索结果是否覆盖 gold evidence；
2. **Answer correctness**：证据和生成器组合的正确率；
3. **Citation completeness**：生成答案中的原子 claim 是否都有支持证据；
4. **Freshness validity**：证据版本是否满足查询时间条件。

使用训练集或 calibration split，针对不同查询类别建立条件模型，并通过校准得到质量下界。

例如：

[
\hat Q(P,q)=f_\theta(P,x(q))
]

再计算：

[
LCB(Q)=
\hat Q(P,q)-r_{1-\delta}
]

其中 (r_{1-\delta}) 来自独立 calibration set 的预测残差。

## 5.5 Progressive Executor

优化器先选择成本最低、可能满足契约的计划。

执行后，系统检查：

* 检索分数间隔；
* 支持证据数量；
* claim-evidence 覆盖；
* verifier 结果；
* 证据时间；
* 剩余延迟预算。

若契约不能被确认满足，则执行补救算子，例如：

* 增大 top-k；
* 增加另一个数据源；
* 调用知识图谱；
* 调用实时 API；
* 换用更强 reranker；
* 换用更强 generator；
* 对缺少引用的 claim 定向检索。

这比一次性预测一个完整 pipeline 更稳健。

---

# 六、公开数据集及用途

## 6.1 HybridQA：表格与文本联合查询

HybridQA 包含超过 7 万个 QA、约 1.3 万张 Wikipedia 表格，每张表平均链接约 44 个文本段落；问题被设计为必须同时使用表格和文本证据。官方代码和数据采用 MIT License。([GitHub][2])

适合测试：

* SQL-first 与 text-first；
* 表格过滤下推；
* EvidenceJoin；
* 结构化聚合；
* 表格—文本计划顺序。

建议把：

* Wikipedia 表格导入 PostgreSQL；
* 链接段落放入 BM25 和向量索引；
* 超链接关系构造成轻量实体图。

它适合做原型和算子正确性实验，但规模不足以单独支撑系统顶会论文。

## 6.2 OTT-QA：大规模开放表格—文本检索

OTT-QA 包含约 4 万多自然语言问题，候选集合包含超过 40 万张表格和约 500 万个文本段落。模型不会直接获得 gold table 和 passage，而是需要从开放候选集合中检索；官方代码和数据采用 MIT License。([OTT-QA][3])

适合测试：

* 大规模计划搜索；
* SQL/table retrieval 与 dense retrieval 顺序；
* BM25 与向量融合；
* 候选数量、延迟和召回权衡；
* 系统吞吐量和 p95/p99 延迟。

它应当是论文的主要规模实验。

## 6.3 CRAG：Web、知识图谱、API 与动态事实

CRAG 提供 4,409 个事实型 QA，并提供模拟 Web 和知识图谱搜索 API。问题覆盖五个领域、八类问题，事实动态性从多年变化到秒级变化。([GitHub][4])

适合测试：

* Web 与 KG API 路由；
* API 延迟和成本；
* 实时数据 freshness；
* 静态索引与实时 API 的选择；
* 动态重新优化；
* 拒答与错误答案成本。

可以为每个 API 设置：

* 延迟分布；
* 单次调用成本；
* 失败率；
* 更新时间；
* 结果可信度。

然后测试优化器是否会根据 freshness contract 改变计划。

## 6.4 HoH：新旧知识共存

HoH 专门研究知识库中新旧信息同时存在时的 RAG。其数据和代码公开，仓库采用 Apache-2.0 License，并基于现实事实变化构造新旧证据冲突。([GitHub][5])

适合测试：

* stale evidence rate；
* timestamp filter；
* 新旧证据冲突；
* freshness-aware rerank；
* 索引滞后；
* 数据更新后的重新优化。

建议把每个事实设置为多个版本：

[
e^{t_1},e^{t_2},\ldots,e^{t_n}
]

通过控制旧版本在向量索引中的比例，构造：

* 10% 旧证据；
* 30% 旧证据；
* 50% 旧证据；
* 最新索引延迟 1、5、30 分钟等环境。

## 6.5 ALCE：引用完整性

ALCE 包含 ASQA、QAMPARI 和 ELI5 三类数据，提供检索结果、生成基线和自动评测代码，评价流畅性、正确性和 citation quality；官方仓库采用 MIT License。([GitHub][6])

适合测试：

* citation completeness constraint；
* inline citation 与 post-hoc citation；
* 引用精确率和召回率；
* 证据不足时的定向补检索；
* 长答案中 claim 数量与检索成本关系。

不要直接使用 ALCE 的 top-100 检索结果作为唯一输入。应重新建立 BM25 和 dense 索引，否则无法测试检索计划优化。

---

# 七、建议构建统一 benchmark：QOptBench

不必重新人工标注大量数据，可以从公开数据构建四个 track。

| Track         | 数据集             | 主要数据源            | 主要约束                  |
| ------------- | --------------- | ---------------- | --------------------- |
| HybridQA-Plan | HybridQA、OTT-QA | SQL、表格、文本、向量     | Quality、Latency       |
| FreshRAG-Plan | CRAG、HoH        | 静态索引、KG、实时 API   | Freshness             |
| CiteRAG-Plan  | ALCE            | BM25、向量、reranker | Citation completeness |
| Dynamic-Plan  | CRAG、HoH 更新流    | 多版本索引、API        | 在线重优化                 |

## 数据存储建议

* PostgreSQL：表格、元数据、版本和实体属性；
* FAISS 或 pgvector：dense retrieval；
* Pyserini/Lucene：BM25；
* PostgreSQL adjacency table 或轻量图数据库：实体关系；
* CRAG mock API：实时外部数据源；
* 本地开源 LLM：生成和验证。

第一版不必部署多个复杂数据库。只要逻辑上支持多源，并能真实测量各算子成本即可。

---

# 八、实验设计

## RQ1：优化器能否满足查询契约？

核心指标：

[
ContractSatisfactionRate
========================

\frac{
#\text{满足全部约束的查询}
}{
#\text{全部查询}
}
]

同时报告：

* quality violation rate；
* citation violation rate；
* freshness violation rate；
* latency SLO violation rate；
* 平均违反幅度。

这是论文最重要的指标，而不是单独报告 EM 或 F1。

## RQ2：满足相同契约时，是否降低成本？

比较：

* token 数量；
* LLM 调用次数；
* API 调用次数；
* GPU 时间；
* 平均成本；
* p50、p95、p99 延迟。

绘制：

[
Cost\text{–}Quality
]

[
Latency\text{–}ContractSatisfaction
]

[
Cost\text{–}CitationCompleteness
]

Pareto 曲线。

## RQ3：跨源执行顺序是否有效？

单独比较：

* SQL-first；
* vector-first；
* graph-first；
* API-first；
* 并行多源；
* 优化器自动选择。

按照 SQL 过滤选择率、问题 hop 数和 freshness requirement 分组报告。

## RQ4：数据更新时是否需要重新优化？

模拟：

* 文档更新；
* 向量索引延迟；
* API 延迟升高；
* API 失败；
* 热点查询变化；
* 查询类型分布变化。

比较：

* 静态最优计划；
* 定期全量重新优化；
* 仅更新 cost model；
* ContractRAG 增量重新优化。

## RQ5：质量估计是否校准？

报告：

* calibration error；
* constraint violation probability；
* predicted LCB 与真实质量关系；
* 不同 domain 上的泛化；
* 训练集到新 workload 的迁移。

## RQ6：计划搜索本身是否可扩展？

改变：

* 算子数量；
* 每个算子物理实现数量；
* 数据源数量；
* 重写规则数量。

报告：

* 优化时间；
* memo 大小；
* 枚举计划数量；
* Pareto 剪枝比例；
* 相对 exhaustive oracle 的 optimality gap。

---

# 九、基线

必须包含以下类型。

## 1. 固定轻量计划

例如：

[
BM25(k=5)
\rightarrow SmallLLM
]

代表低成本生产配置。

## 2. 固定重型计划

例如：

[
BM25+Dense+Graph/API
\rightarrow LargeReranker
\rightarrow Verifier
\rightarrow LargeLLM
]

代表“尽可能高质量”的配置。

## 3. Query Router

从若干固定 pipeline bundle 中选择一个，但不修改算子顺序。

用于证明你的贡献不是简单的分类路由。

## 4. Bayesian/Grid Pipeline Optimization

在固定 pipeline 结构中优化 top-k、模型和检索参数。

用于证明结构化计划搜索优于黑盒调参。

## 5. Abacus-RAG

将 RAG pipeline 表示为 Abacus 语义算子，尽可能公平地加入相同物理实现。

这是最重要的系统基线。Abacus 已经覆盖质量、成本和延迟的约束优化，如果不与它比较，SIGMOD/VLDB 审稿人很可能直接质疑工作增量性。

## 6. Oracle

在小型计划空间中 exhaustive execution，得到真正最优计划。

用于评价：

[
OptimalityGap=
\frac{
Cost(P_{\text{ours}})
---------------------

Cost(P_{\text{oracle}})
}{
Cost(P_{\text{oracle}})
}
]

---

# 十、必要消融实验

至少做：

* 去掉 approximate rewrite risk；
* 去掉 citation constraint；
* 去掉 freshness constraint；
* 去掉 runtime reoptimization；
* 去掉 progressive execution；
* 只使用平均质量，不使用 LCB；
* 全局 cost model 与 query-conditioned cost model；
* 无 Pareto pruning；
* 无 source-order optimization。

其中最关键的是：

> **只选择模型和 top-k，与完整算子顺序优化的比较。**

否则审稿人会认为整套系统最终仍只是自动调参。

---

# 十一、一个完整查询例子

假设问题是：

> 截至 2024 年，表中获得某奖项的导演中，出生国家人口最多的是谁？

涉及：

1. 从表格中找到获奖导演；
2. 从文本中获取出生地；
3. 从实体图或 API 获取出生国家；
4. 查询该国家截至 2024 年的人口；
5. 排序；
6. 为导演、出生地和人口三个 claim 提供引用。

候选计划一：

[
DenseRetrieve(all)
\rightarrow LLM
]

成本高、引用覆盖不稳定，而且可能检索到过期人口数据。

候选计划二：

[
SQLFilter(award)
\rightarrow TextRetrieve(birthplace)
\rightarrow KGJoin(country)
\rightarrow API(population,2024)
\rightarrow Sort
\rightarrow Generate
]

当 freshness 要求严格时，选择计划二。

如果用户允许人口数据延迟一年，优化器可以改用本地 KG：

[
SQL
\rightarrow Text
\rightarrow LocalKG
\rightarrow Generate
]

这正是“查询契约改变物理执行计划”的直观案例。

---

# 十二、最小可行版本

第一阶段不要立即实现全部算子。

## MVP 数据集

* HybridQA；
* CRAG；
* ALCE。

## MVP 数据源

* PostgreSQL；
* BM25；
* FAISS/pgvector；
* CRAG mock API。

## MVP 算子

[
SQLRetrieve
\rightarrow
Sparse/DenseRetrieve
\rightarrow
Join
\rightarrow
Rerank
\rightarrow
Verify
\rightarrow
Generate
\rightarrow
CiteCheck
]

## MVP 约束

只保留四个：

[
Quality,\ Citation,\ Freshness,\ Latency
]

成本作为优化目标。

## 第二阶段扩展

再加入：

* OTT-QA 大规模实验；
* HoH 更新流；
* GraphExpand；
* 在线 workload drift；
* 增量重新优化。

---

# 十三、论文可以声称的核心创新

在真正完成系统后，贡献可以写成：

1. **提出 evidence contract**，统一表示 RAG 的答案质量、引用完整性、新鲜度与延迟要求。
2. **提出 Hybrid RAG algebra**，支持 SQL、稀疏检索、向量检索、图扩展和实时 API。
3. **提出带误差预算的近似计划等价理论**，允许优化器安全地使用非严格等价的 RAG 重写规则。
4. **提出 Contract-Cascades**，在成本、延迟、质量下界、引用下界和 freshness 上搜索 Pareto 计划。
5. **提出渐进式执行和在线重新优化机制**，在数据、索引和 API 状态变化时维持查询契约。
6. **基于公开数据构建 QOptBench**，覆盖 table-text、动态事实、Web/KG API 和 citation-intensive RAG。

其中第 3 点最可能成为真正的算法创新，第 4、5 点构成系统创新，第 6 点构成实验和社区价值。

---

# 十四、客观投稿判断

## SIGMOD / VLDB

适合，但必须满足：

* 有正式 algebra；
* 有计划转换规则；
* 有真实执行引擎；
* 有 Abacus 对比；
* 有百万级以上候选或 OTT-QA 规模实验；
* 有 dynamic workload；
* 不只报告 QA 准确率；
* 提供优化开销、吞吐和尾延迟。

做到这些，属于有竞争力的数据库系统论文。

## AAAI

AAAI 版本应把重点改成：

> Risk-calibrated RAG planning under quality and citation constraints。

弱化数据库执行引擎，强化：

* 质量概率估计；
* 约束满足；
* 在线决策；
* 分布外泛化；
* 不确定性校准。

但从整体构思看，这个方向更天然地适合 **SIGMOD/VLDB**。

## 最终建议

不要把论文核心写成：

> 在 quality、cost、latency 之间自动寻找最优 RAG 配置。

这已经过于接近 Abacus 和 RAG-Stack。

应写成：

> **在动态异构数据环境中，根据证据质量、引用完整性和新鲜度契约，对跨 SQL、向量、图和 API 的近似 RAG 执行计划进行风险可控的优化与在线重规划。**

这个表述才具有明确、可防守且较本质的创新边界。

[1]: https://arxiv.org/abs/2510.20296?utm_source=chatgpt.com "RAG-Stack: Co-Optimizing RAG Quality and Performance From the Vector Database Perspective"
[2]: https://github.com/wenhuchen/HybridQA "GitHub - wenhuchen/HybridQA: Dataset and code for EMNLP2020 paper \"HybridQA: A Dataset of Multi-Hop Question Answeringover Tabular and Textual Data\" · GitHub"
[3]: https://ott-qa.github.io/?utm_source=chatgpt.com "OTT-QA"
[4]: https://github.com/facebookresearch/CRAG/ "GitHub - facebookresearch/CRAG: Comprehensive benchmark for RAG · GitHub"
[5]: https://github.com/0russwest0/HoH?utm_source=chatgpt.com "\"HOH: A Dynamic Benchmark for Evaluating the Impact of ..."
[6]: https://github.com/princeton-nlp/ALCE "GitHub - princeton-nlp/ALCE: [EMNLP 2023] Enabling Large Language Models to Generate Text with Citations. Paper: https://arxiv.org/abs/2305.14627 · GitHub"
