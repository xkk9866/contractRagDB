# RAG / 数据系统方向 CCF-A（VLDB/SIGMOD 风格）论文写作指南

基于对 7 篇已录用 PVLDB 论文的逐篇分析：Abacus (p1060)、DocETL (p3035)、
LOTUS (p4171)、KEN (p902)、Semantic Integrity Constraints (p4073)、
LEGO-GraphRAG (p3269)、Chai (p4560)。

## 1. 整体叙事脉络（story arc）

所有样本共享同一个五拍结构：

1. **大趋势一句话**：LLM/语义算子正在改变数据处理（一段，不超过 4 句，
   带 10+ 条引用展示领域活跃度）。
2. **具体化的痛点**：立刻落到一个具体、可感的场景。DocETL 用
   Example 1.1（警察不当行为调查），KEN 用 PDF 发票解析，Abacus 用
   文献检索 pipeline。**痛点必须能用一句话对非专家复述**。
3. **"现有系统为什么不行"**：不是罗列缺点，而是指出一个*结构性*缺陷
   （DocETL："它们假设用户算子按原样执行就够准"；LOTUS："要么无保证、
   要么只支持批推理原语"；KEN："只有粗粒度实现可选"）。
4. **Our approach / key insight**：用 2–3 个"关键想法"段落
   （Abacus 的 "three key ideas"），每个想法一段，先说难点再说机制。
5. **Performance 段 + 贡献列表**：具体数字（"10.8x cheaper and 3.4x
   faster"），贡献 3–4 条 bullet，每条挂 Section 号。

## 2. 语气与句式

- **自信的陈述句，present tense**："Abacus estimates operator
  performance by..."、"We present..."。避免 hedging（"we try to",
  "somewhat"），但对局限诚实（DocETL 明说 "finding optimal pipelines
  is impossible... however..."）。
- **短句开头、长句展开**。段首句是全段主旨（topic sentence），审稿人
  只读段首句也能重建论文骨架。
- **对比修辞**是数据库社区的标志性句式："Like a relational query, ...;
  unlike a relational query, ..."；"cardinality estimates pick the
  plan, the executor's correctness does not depend on them"。
- **命名法**：系统名 smallcaps；概念第一次出现用 *italics* 并立即给
  定义；之后一律用同一个词（不换同义词）。
- 数字规则：正文中的比较数字保留 1–2 位有效（"6.7%–39.4% better"），
  表格里 2–3 位小数；每个正文数字必须能在某张表/图中找到。

## 3. 引言的微观结构（以 DocETL/KEN 为模板）

- ¶1 大趋势 + 引用簇。
- ¶2 Example X.Y（编号的运行示例，楷体人名/场景，贯穿全文复用）。
- ¶3–4 为什么"直接做"会失败：列出 2–3 个具体失败模式，每个配一句
  文献支撑。
- ¶5 "However, we cannot expect a user to ..."：把负担从用户转到系统，
  引出自动化需求。
- ¶6 We present X：一句话定位 + 2–3 个机制概览，每个机制一个粗体词。
- ¶7 Performance 数字段。
- ¶8 贡献 bullet（挂 Section 号）。

## 4. 章节组织

- **VLDB 惯例：Related Work 放最后**（DocETL §6、Abacus §5、KEN 末尾），
  引言里用一段"现有工作不行"承担前置对比职责。若方法依赖大量背景
  （统计检验），单设 §2 Background/Overview（Abacus 的 §2 System
  Overview：先 end-to-end 流程，再指出"两个使之可行的关键算法在 §3"）。
- **System Overview 章**：图 + 输入/输出 + 逐步 walk-through，
  "对应 Figure 2 的第 (3) 步"这类锚定句非常常见。
- 每个设计章节的开头一句话回答"本章解决前一章留下的什么问题"。
- 实验章以 RQ 列表开头（LEGO-GraphRAG、Abacus 都是），每个小节
  标题即 RQ；每个实验小节末尾有一句 takeaway（可加粗）。

## 5. 图的设计

- **Figure 1 是论文的电梯演讲**：不是架构图，而是"一个具体输入 →
  两种不同目标下的两个输出"的对比图（Abacus Fig.1：同一程序、
  MaxQuality 与 Cost<$1 两个物理计划并排；KEN Fig.1：左 FLOPs-精度
  散点、右反例 runtime 条形，一张图同时给出 promise 和 pitfall）。
- 架构总览图（Figure 2）：从左到右的数据流，用户输入在左，最终
  产物在右，模块用颜色区分职责；caption 是完整段落，独立可读
  （"The developer provides ...; Abacus (1)...(2)...(3)...(4)"）。
- 实验图：一行多个小 panel、共享 y 轴；对比方法用固定颜色/marker
  贯穿全文；预算线/约束线用红色虚线；caption 里点明结论而不只是
  描述坐标轴。
- caption 全部是完整句子，加粗开头短语可选；正文引用图时告诉读者
  看什么（"as shown on the right of Figure 1, ... takes longer
  even though it saves FLOPs"）。

## 6. 表的设计

- 表 caption 放顶部（ACM 格式），说明协议 + 一句结论
  （"ContractRAG is the only optimizer meeting the budget"）。
- booktabs 三线表；分组用 \midrule；最优值加粗；
  次优可加下划线；oracle/skyline 单独隔开。
- 每张表在正文中至少被"读"一次：挑 2–3 个单元格讲故事，
  不复述整表。
- 符号表（LOTUS/Abacus 的 operator 表）：让形式化定义扫一眼可查。

## 7. 定理与统计内容的呈现（LOTUS/本文相关）

- 保证以"接口"呈现：定理陈述用户可感的性质（"for any number of
  candidates, any score quality"），证明进附录，正文给 3–4 行
  proof sketch。
- 每条假设（assumption）必须有一段"这个假设在真实负载上被违反时
  会怎样 + 我们实测了它"——这是数据库审稿人和 ML 审稿人都买账的
  写法（本文 Assumption NSH 的处理就是对的）。
- 把统计概念翻译成数据库概念的"字典句"：sufficiency score ≈
  cardinality feedback，voucher ≈ cost model，certification ≈
  执行引擎正确性契约。这类句子是 VLDB 论文的加分项。

## 8. 实验章写法

- 开头 RQ 编号列表；Setup 小节交代 workload/baseline/指标，
  baseline 的适配方式要一句话说清（"adapted to our ladders ---
  a text classifier trained to predict the cheapest correct rung"）。
- 主结果表 + "Three observations. (i)...(ii)...(iii)" 的枚举式解读。
- 必须有的成分：消融（每个机制单独关掉）、开销测量（优化器自身
  成本）、可扩展性/敏感性扫描、失败模式/诚实的负结果
  （DocETL 明写哪个任务提升有限及原因）。
- 数字前后一致性是 desk-reject 级别的问题：摘要、引言、正文、
  表格四处的同一指标必须逐字核对。

## 9. 用词表（样本中高频且有效）

- 机制动词：orchestrate, materialize, enumerate, escalate,
  amortize, expose (a trade-off), navigate (a trade-off)。
- 定位名词：formalism, abstraction, interface, discipline,
  division of labor, skyline。
- 转折句式："However, X alone is not enough: ..."；
  "This is precisely why ..."；"The critical difference is that ..."。
- 避免：exploit（用 leverage/use）、novel 滥用（贡献列表外少用）、
  "very/really/significantly" 无数字支撑时不用。
