# ContractRAG 研究设计（v1，2026-07-13）

## 0. 一句话定位

> 在异构（表格/文本/向量/KG-API）动态数据上，把 RAG 查询规划形式化为
> **“约束可行性只能通过校准统计检验”的成本最优化问题**，
> 用分布无关的风险控制（Learn-Then-Test / e-process）证书驱动
> 计划选择、近似重写与渐进执行三个层面，
> 给出有限样本契约满足保证，同时把期望成本压到远低于任何固定管线。

与三条已有研究线的本质区别：

| 研究线 | 代表 | 缺失 |
|---|---|---|
| 语义算子优化器 | Abacus (VLDB'26), RAG-Stack | workload 级单计划、无统计保证、无逐查询自适应、无 freshness/citation 契约 |
| 自适应路由 | Adaptive-RAG, X-Router, CA-RAG, RouteRAG | 启发式，违约率不可控，无理论 |
| conformal RAG | TRAQ, C-RAG, Conformal-RAG | 固定管线，不做计划优化，不省成本 |

ContractRAG 是这三条线的**公共上界**：以统一的风险预算贯穿
(i) 计划格上的证书化选择，(ii) 带 (ε,δ) 凭证的近似重写，(iii) 校准的渐进执行升级。

## 1. 形式化

### 1.1 契约
查询分布 D 上，契约 C = ⟨τ_q, τ_c, Δ_f, B_l, α, δ⟩：
- 质量：P(quality(π(q)) < τ_q) ≤ α_q
- 引用：P(citation-recall(π(q)) < τ_c) ≤ α_c
- 新鲜度：max age(e) ≤ Δ_f（构造性硬约束）
- 延迟：p95 latency ≤ B_l
- δ：证书本身的失效概率（对校准集随机性）

**契约满足证书**：策略 π 满足 C 当且仅当基于 n 个校准样本的
Hoeffding–Bentkus p 值检验拒绝 H0: R(π) > α，FWER 由固定序列检验控制在 δ。

### 1.2 计划格与近似重写
逻辑计划由算子代数生成：Retrieve_src(k) / Fuse / Rerank(m) / Verify / Generate(model) / CiteFix。
重写规则 R_i: P → P'（成本↓），携带**风险凭证** (ε_i, δ_i)：
P(ΔLoss(R_i) > ε_i) ≤ δ_i（校准集上二项上界证书）。
组合定理：串联重写凭证按并集界累加（可证但松）；
**直接证书**（对复合计划整体做 LTT）严格更紧 → 定理+实验验证 tightness gap。

### 1.3 渐进执行（核心成本来源）
成本升序计划梯子 P_0 ≺ P_1 ≺ ... ≺ P_L（由重写规则从满计划生成）。
每级 rung j 产生运行时充分性得分 s_j(q)（检索边际、验证器支持率、
引用覆盖率、新鲜度校验、弃答信号）。
策略 π_λ：在第一个 s_j(q) ≥ λ_j 的 rung 停止，否则升级；最后一级必答。
λ 通过 1 维单调路径参数化 + 固定序列 LTT 校准：
从最保守 λ 出发按期望成本降序走，保留 p 值 ≤ δ 的最便宜策略。
⇒ **定理 1**：返回的 π_λ 以概率 ≥ 1−δ（对校准集抽样）满足 R(π_λ) ≤ α。
组条件版本：按查询组（如 CRAG dynamism、问题类型）分别校准
⇒ 组条件契约（比 marginal 更强，回应"平均质量掩盖尾部"批评）。

### 1.4 在线契约监控（动态部分）
部署后用 e-process（test supermartingale，Ville 不等式）监控违约指示序列：
E_t = ∏ (1 + w·(ℓ_i − α))，E_t ≥ 1/δ 触发重新校准/升级默认 rung。
⇒ **任意时刻有效**（anytime-valid）的契约保障，覆盖 workload drift、
索引陈旧化、API 退化三类漂移。这是数据库"在线重优化"的统计化版本。

## 2. 系统

- **算子层**：DuckDB(SQL/表)、bm25s(稀疏)、FAISS+BGE(稠密)、CRAG mock KG-API、
  bge-reranker(GPU)、MiniCheck/NLI 验证器(GPU)、DashScope qwen 阶梯
  (flash→plus→max) 生成、ALCE 式 CiteCheck。
- **执行引擎**：每次执行记录 (tokens, ¥cost, wall-latency, evidence ages, scores)。
- **缓存**：所有 LLM 调用/检索结果 SQLite 缓存（可复现+省钱）。
- **并行**：32-48 线程 DashScope 并发。

## 3. 实验 Track

| Track | 数据 | 源 | 契约 | 规模 |
|---|---|---|---|---|
| A HybridQA | HybridQA + WikiTables | 表(SQL)+文本(BM25/Dense) | 质量+延迟 | cal 500 / test 1000+ |
| B CRAG | CRAG task1&2 + mock KG API | web页+KG API | 质量+新鲜度(组条件) | cal ~500 / test ~1300 |
| C ASQA | ALCE-ASQA | BM25/Dense/rerank | 引用完整性+质量 | cal 300 / test 648 |

## 4. 基线
1. 固定轻量 / 2. 固定重型（每 rung 单独作为固定计划，构成成本-质量锚点）
3. Adaptive-RAG 式复杂度路由（复现）
4. CA-RAG 效用路由（复现）
5. Optuna 贝叶斯 workload 级调参
6. Abacus workload 级 Pareto-MAB 优化（palimpzest 或忠实复现）
7. TRAQ/conformal 固定管线
8. Oracle（子集上穷举）

## 5. 指标
- 契约满足率 CSR、各约束违约率、违约幅度
- ¥成本、tokens、LLM 调用数、p50/p95 延迟
- Cost–Quality / Cost–Citation Pareto 曲线
- 校准图：目标 α vs 实测违约率（跨 α 扫描）
- 组条件违约率（dynamism / 问题类型 分组）
- 重写凭证 vs 直接证书 tightness
- 漂移流实验：违约率时间序列 + e-process 触发点
- Oracle optimality gap

## 6. 里程碑
M1 数据+索引 → M2 算子+引擎 → M3 梯子执行数据收集(校准+测试) →
M4 校准器+基线 → M5 主表+消融 → M6 漂移实验 → M7 论文
