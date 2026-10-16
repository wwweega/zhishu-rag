# 知枢 Zhishu · 企业级 RAG 知识中枢

> 开发周期：2026.01 – 2026.05
> 面向企业私有文档的检索增强问答系统：**解析 → 分块 → 向量化 → 混合检索 → 重排 → 生成治理 → 引用溯源**。
>
> 除了把答案找出来，重点解决三个生产环境的实际问题：引用可溯源、答不了就拒答、慢在哪能说清。

<p align="left">
  <a href="https://github.com/wwweega/zhishu-rag/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/wwweega/zhishu-rag/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 一、这个项目解决什么问题

企业内部的规章制度、采购流程、售后条款散落在 PDF / Word / 扫描件里，员工想查一条"年假能顺延到什么时候""5 万的设备谁审批"，往往要翻半天文件。

直接丢给大模型会编造；用通用搜索又答不出自然语言问题。知枢的思路是：

1. **答得准**：向量语义召回 + BM25 关键词召回 + RRF 融合 + Cross-Encoder 重排，兼顾"召得全"和"排得准"；
2. **答得可溯源**：每一句结论都带 `[证据N]`，且引用可**机器校验**（逐字比对原文），杜绝幻觉引用；
3. **答不了就拒答**：知识库里没有的信息，明确拒答而不是编造 —— 这是企业场景的底线。

---

## 二、5 分钟跑起来

```bash
# 1. 安装依赖（Python 3.11）
pip install -r requirements.txt

# 2. 配置（复制后填自己的 Key；不填也能跑，自动降级 mock 模式）
cp .env.example .env
#   LLM_API_KEY / LLM_BASE_URL / LLM_MODEL      ← 阿里云百炼 / DeepSeek / 任意 OpenAI 兼容接口
#   EMBED_MODEL / RERANK_MODEL                  ← 嵌入与重排模型
#   完全没有 Key 也没关系：会降级 hash 向量 + mock 生成，全链路照样跑通

# 3. 一键端到端演示（入库 → 检索 → 生成 → 引用校验 → 缓存验证）
python -m scripts.demo

# 4. 跑测试（34 个，全部走降级后端，无需任何外部服务）
python -m pytest tests/ -q

# 5. 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
#    演示前端： http://localhost:8080
#    接口文档： http://localhost:8080/docs
```

**Docker 一键起全套依赖（Milvus + PG + Redis + 应用）：**
```bash
docker compose up -d --build
docker compose logs -f app
```

> 本项目**每一步都设计了降级**：没有 Milvus 就用 numpy 暴力检索，没有 PG 就用 SQLite，
> 没有 Redis 就用进程内 LRU，没有模型 Key 就用 mock。这样无外部依赖也能把整个链路跑起来复现。

---

## 三、架构与数据流

```
                            ┌──────────────── 入库链路（离线） ────────────────┐
   PDF/Word/MD/扫描件 ──▶ 文档解析 ──▶ 语义分块 ──▶ 父子块构建 ──▶ 向量化 ──▶ 存储
   (MinerU/PaddleOCR/        (版面/表格)   (带 overlap)   (child 检索         (Milvus)
    PyMuPDF 三级降级)                                   parent 生成)        (PG 元数据)

                            ┌──────────────── 问答链路（在线） ────────────────┐
   用户提问 ──▶ 查询改写 ──▶ 混合召回 ─────────▶ RRF 融合 ──▶ 重排 ──▶ 父块回溯 ──▶ 生成治理 ──▶ 引用校验
              (改写+HyDE)   (向量 + BM25)       (排名倒数融合)  (Cross-Encoder) (small→big)   (强制JSON    (逐字比对
                                                                                              + 拒答)      + 编号自愈)
```

**为什么是这套组合（常见疑问）**：

| 设计选择 | 为什么 | 不加会怎样 |
|---|---|---|
| **混合检索**（向量 + BM25） | 向量擅长语义，BM25 擅长编号/型号/条款号等精确匹配 | 只靠向量：查"第三条""GB/T 1234"会翻车 |
| **RRF 融合**（非加权求和） | 只用排名不用分数，天然跨量纲，免去分数归一化调参 | 加权求和要反复调权重，换个模型就失效 |
| **Cross-Encoder 重排** | 召回求全、重排求准，一次精读 top-N | 直接拿向量 top1 喂模型，精度差一截 |
| **父块 small-to-big** | 小块召回准、大块上下文全 | 只喂小块：断章取义；只喂大块：召回差 |
| **引用机器校验** | 把"引用"从口头承诺变成可验证事实 | 用户"看到引用"却点不开、对不上 |
| **证据不足拒答** | 没依据时敢于说"不知道" | 一本正经地编造 |

---

## 四、关键技术点（每个都有对应测试）

| 模块 | 关键点 | 文件 |
|---|---|---|
| 解析 | 三级降级：MinerU（版面/表格）→ PaddleOCR（扫描件）→ PyMuPDF（文本层）；**先按后缀分发**（`.md` 走文本解析器，避免误送 PDF 解析器） | `app/ingest/parsers.py` |
| 分块 | 语义分块 + 重叠窗口；父子块构建；**先定父块 id 再回填 child.parent_id**（否则父块回溯失效） | `app/ingest/chunker.py` |
| 向量库 | Milvus HNSW(IP) / numpy 暴力检索双后端；**delete 前必须 load**（Milvus 老坑）；**只对 child 建向量** | `app/storage/vector_store.py` |
| 缓冲 | 键含**知识库版本号**防脏读；版本号存 **Redis 而非进程内存**（多进程/重启都不失效） | `app/storage/cache.py` |
| 检索 | jieba 分词保留字母数字；RRF 融合；查询改写 + HyDE；过滤条件下推 Milvus expr | `app/retrieval/hybrid_retriever.py` |
| 生成治理 | 强制 JSON 输出；句级 `[证据N]`；**引用编号自愈**；空 quote 兜底；**归一化拒答阈值** | `app/generation/rerank_generate.py` |
| 可观测 | Trace/Span 树；**墙钟 = 区间并集**；**阶段耗时也取并集**（防嵌套重复求和）；P50/P95；按 token 计费 | `app/core/observability.py` |
| 评测 | Faithfulness / Answer Relevancy / Context Recall / LLM-as-Judge；**CI 门禁按单项下限**卡 | `app/eval/harness.py` |

---

## 五、实测结果（真实跑出来的，非编造）

> 以下数据来自本机**真实**运行：Milvus v2.4.17 + PostgreSQL 16 + Redis 7 容器，
> LLM 走阿里云百炼（qwen-plus + text-embedding-v3 + gte-rerank-v2）。

**入库**：2 篇制度文档 → 每篇 6 个 child + 2 个 parent。

**问答**（节选自 `python -m scripts.demo`）：

| 问题 | 回答 | 引用 |
|---|---|---|
| 年休假可以顺延到什么时候？ | 最多顺延至次年 3 月 31 日 | ✓ 证据1，已核对原文 |
| 采购 5 万元设备谁审批？ | 须经总经理审批 | ✓ 证据1，已核对原文 |
| 迟到几次影响绩效？ | 每月超 3 次（每次超 30 分钟）不得评优 | ✓ 证据1，已核对原文 |
| 满 12 年工龄，年假几天+能否顺延？ | 10 天；可顺延至次年 3 月 31 日 | ✓ 证据1（多跳合成） |
| 年会抽奖预算多少？（库外） | **拒答**：证据未提及 | —（正确拒答） |

**质量门禁**（`python -m scripts.run_eval --gate`）：

```
faithfulness        1.0     answer_relevancy    1.0
context_recall      1.0     llm_judge           0.7
拒答准确率          2/2     平均耗时            6.9s   P95 9.1s
单次问答成本        ≈ $0.0009
```

**单测**：`34 passed`。

---

## 六、工程化踩坑记录（开发者常被问到的"你遇到过什么问题"）

这个项目在开发过程中真实修复了一批**"跑通≠跑对"**的隐性 bug，每个都留了回归测试：

1. **`marshmallow 4.x` 删了 `__version_info__`** → 拖垮 `pymilvus` 依赖链，锁回 3.x。
2. **`.md` 被送去 PyMuPDF 解析** → 报错后静默产出 0 个块。改为**按后缀分发**解析器。
3. **Milvus 删除时报 "collection not loaded"** → 删除前必须先 `load()`。
4. **`build_parent_child` 从没填充 `children`** → 检索静默退化（不报错，但质量掉）。补上显式 append。
5. **分块重叠窗口逻辑写反** → 整篇文档被并成一个巨型块。重写为显式重叠上限。
6. **父块 id 后置改写** → child 的 `parent_id` 指向失效的旧 id，父块回溯失灵。改为**先定父 id 再回填子**。
7. **父块也被向量化** → 和大块在同一空间竞争、语义被稀释。改为**只对 child 建向量**。
8. **缓存版本号用 `int(time.time())`** → 同一秒内批量入库产生相同版本号，缓存不失效。改为单调计数器。
9. **版本号放进程内存** → 多进程/重启后旧缓存重新可见（脏读）。改为存 **Redis**。
10. **空知识库时仍调用模型** → 会凭空编造。改为**调用前硬拒答**。
11. **`verified=false` 假阴性** → 证据原文在两处登记，口径不一致。改为**产生编号处即登记原文**（单一事实来源）。
12. **模型把编号写进正文却不填 `citations`** → 引用校验 0/0 通过但实际失效。改为**从正文正则回填编号**。
13. **`stage_breakdown` 对嵌套 span 重复求和** → 阶段耗时（10690ms）**超过**端到端墙钟（5950ms）。改为按阶段取**时间区间并集**。
14. **拿 rerank 绝对分比固定阈值判拒答** → 百炼 gte-rerank 对真正相关的块也只给 ~0.19 分，**假拒答**。改为**批内归一化 + 相对阈值**。
15. **`expand_parent` 用父块文本覆盖 child** → 引用指向放大后的父块、且诱使模型按位置乱标编号。改为 **child 保留原文（可引用）、父块另存 `parent_text`（做语境）**。

> 这些坑的共同点：**不报错，但结果变差**。所以本项目的原则是"**跑通不算数，要跑对**" ——
> 每一处修复都配一个回归测试（见 `tests/test_core.py`）。

---

## 七、目录结构

```
zhishu-rag/
├── app/
│   ├── core/          # 配置、LLM 客户端（含重试/降级/错误分类）、可观测
│   ├── storage/       # 向量库（Milvus/numpy）、元数据（PG/SQLite）、缓存（Redis/LRU）
│   ├── ingest/        # 解析器（MinerU/PaddleOCR/PyMuPDF/Text）、父子块切分
│   ├── retrieval/     # BM25、RRF 融合、查询改写、HyDE、混合检索
│   ├── generation/    # 重排、父块回溯、生成治理、引用校验
│   ├── eval/          # 评测 harness + CI 门禁
│   ├── api/           # FastAPI 路由（含 SSE 流式）
│   ├── pipeline.py    # 门面：ingest / ask / stats
│   └── main.py        # 应用入口
├── tests/             # 34 个测试，全部降级模式可跑
├── scripts/           # demo / ingest / run_eval / reproduce
├── frontend/          # 单文件演示 UI（SSE 流式 + 引用高亮）
├── data/corpus/       # 示例语料
├── docs/              # 架构设计 / 制作过程
├── docker-compose.yml # Milvus + etcd + MinIO + PG + Redis + app
└── Dockerfile
```

---

## 八、诚实说明（边界与局限）

- **示例语料只有 2 篇**，用于演示链路正确性；真实企业场景需接入自有文档并重新评测。
- **评测集 10 题**（含 2 个拒答负例），是**功能性验证**而非统计意义上的基准；
  指标数值仅代表"在这套小语料上链路是对的"，**不代表**在任意语料上的泛化能力。
- **未做鉴权/多租户**：本仓库聚焦 RAG 链路本身，生产化需要的权限体系、审计、限流不在此范围。
- 依赖 Key 的能力（真实嵌入/重排/生成）需要自备；无 Key 时自动降级，但降级模式的效果**不可用于评估质量**。

---

## 九、License

[MIT](LICENSE)

---

## 十、延伸阅读

- [制作过程：背景、选型与核心难点](docs/制作过程.md)
- [架构设计详解](docs/架构设计.md)
