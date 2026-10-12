"""
知枢 · 检索层（混合检索：BM25 稀疏 + 向量稠密 + RRF 融合）
============================================================
简历原文：「BM25 稀疏 + 向量稠密双路召回 + RRF 融合，再经 bge-reranker-v2 重排」

为什么必须混合检索：
  · 纯向量（稠密）：擅长"语义相近"——问"怎么请年假"能召回"休假申请流程"。
     但它对**专有名词、型号、编号、条款号**极差：
     问"GB/T 19001-2016 第 8.5.2 条"这种，向量会召回一堆"看起来像"的内容，
     真正的条款反而排不上号。
  · 纯 BM25（稀疏）：擅长**精确词匹配**——编号、型号、术语一打一个准。
     但它不理解同义（"年假" 搜不到 "休假"）。
  → 两者互补。真实企业文档里既有"语义型问题"也有"编号型问题"，
    所以混合检索是标配而非锦上添花。

RRF 融合（Reciprocal Rank Fusion）：
  score = Σ 1 / (k + rank_i)
  设计要点：为什么用 RRF 而不是"分数加权求和"？
    → BM25 分和余弦相似度**量纲完全不同**（前者可到几十，后者在 0~1）。
      直接加权必须做归一化，而归一化方式（min-max / z-score）本身引入超参且不稳定。
      RRF 只用**排名**，天然跨量纲、无需调参、对异常分鲁棒 —— 这就是它的价值。
      k=60 是原论文经验值，作用是"压低头部排名的统治力"，让中段排名也能贡献。

过滤（权限 / 时效）：
  元数据过滤必须在**召回阶段下推**（Milvus 的 expr），而不是召回后再筛 —— 否则
  你召回了 10 条全是无权访问的，筛完剩 0 条，白跑一趟。
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass

import jieba

from app.core.config import settings
from app.core.llm_client import llm
from app.core.observability import TraceRecorder
from app.ingest.chunker import Chunk, estimate_tokens
from app.storage.meta_store import get_meta_store
from app.storage.vector_store import VectorStore

logger = logging.getLogger("zhishu.retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str            # vector | bm25 | hybrid | rerank
    parent_id: str | None = None
    meta: dict | None = None
    rank_vector: int | None = None
    rank_bm25: int | None = None


# ============================================================
# BM25 稀疏检索
# ============================================================
class BM25Index:
    """
    轻量 BM25（rank_bm25）。设计要点：为什么不用 Elasticsearch？
      → ES 当然更适合生产（分布式、倒排持久化、分词器生态），但本项目是
        "可私有化一键部署"，多一套 ES 会显著提高部署门槛。
        在当前规模（万级块）下，内存 BM25 的延迟完全可接受。
        真要上量，接口不变，换 ES 实现即可 —— 这就是把检索抽象成接口的意义。
    """

    def __init__(self) -> None:
        self.corpus_tokens: list[list[str]] = []
        self.chunk_ids: list[str] = []
        self._bm25 = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        中文分词用 jieba；英文/数字按词切并转小写。
        设计要点：为什么分词这么关键？
          → BM25 是词袋模型，"年假" 若被切成 "年"/"假" 两个词，
            精确匹配能力就废了。分词质量直接决定稀疏检索的上限。
        """
        text = text.lower()
        words: list[str] = []
        for tok in jieba.cut(text):
            tok = tok.strip()
            if not tok:
                continue
            if re.fullmatch(r"[a-z0-9\.\-/]+", tok):     # 英文/型号/编号 原样保留
                words.append(tok)
            elif re.fullmatch(r"[\u4e00-\u9fff]+", tok):  # 中文词
                words.append(tok)
        return words

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi
        self.corpus_tokens = [self.tokenize(c.text) for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        if self.corpus_tokens:
            self._bm25 = BM25Okapi(self.corpus_tokens)
        logger.info("BM25 索引构建完成：%d 篇", len(self.chunk_ids))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []
        q = self.tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]


# ============================================================
# 查询改写 & HyDE
# ============================================================
REWRITE_PROMPT = """你是企业知识库的检索助手。请把用户的口语化问题改写成更适合检索的关键词组合。
要求：
1) 补全同义词与专业术语（如"年假"→"年休假 带薪年假 请假流程"）
2) 保留原文中的编号、型号、条款号（这些必须原样保留）
3) 只输出改写后的检索串，不要解释、不要引号

用户问题：{q}
检索串："""

HYDE_PROMPT = """请针对下面的问题，写一段"假设性的答案"（100 字以内）。
重点：这段答案不会展示给用户，只用于向量检索——它越像真实文档里的措辞，检索越准。

问题：{q}
假设答案："""


def rewrite_query(query: str, trace: TraceRecorder | None = None, span=None) -> str:
    """
    查询改写：口语 → 检索词。
    设计要点：为什么改写有效？
      → 用户说"我去年没休完的假能顺延吗"，文档里写的是"年休假结转规则"。
        字面词重叠为 0，向量也未必拉得近。改写后补上"年休假 结转"，
        两路召回都能命中。
    降级：LLM 不可用时直接用原问题（绝不因改写失败而让整个检索挂掉）。
    """
    if not settings.use_query_rewrite:
        return query
    try:
        with (trace.span("query_rewrite", kind="retrieve") if trace else _null_span()) as sp:
            res = llm.chat(REWRITE_PROMPT.format(q=query), temperature=0.0, max_tokens=128,
                           trace=trace, trace_span=sp)
            out = res.text.strip().strip('"').strip()
            return out if out and len(out) < 200 else query
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询改写失败，用原问题：%s", exc)
        return query


def hyde_expand(query: str, trace: TraceRecorder | None = None, span=None) -> str | None:
    """
    HyDE（Hypothetical Document Embeddings）：
    让模型先"编"一段假设答案，再用这段答案去检索向量库。
    原理：假设答案的用词更接近真实文档，能显著缩小 query/document 的语义鸿沟。
    诚实说明：HyDE 会多一次 LLM 调用（成本 + 延迟），
               所以做成开关（settings.use_hyde），并在评测里对比开关效果。
    """
    if not settings.use_hyde or llm.mock:
        return None
    try:
        with (trace.span("hyde", kind="retrieve") if trace else _null_span()) as sp:
            res = llm.chat(HYDE_PROMPT.format(q=query), temperature=0.3, max_tokens=200,
                           trace=trace, trace_span=sp)
            return res.text.strip() or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("HyDE 失败，跳过：%s", exc)
        return None


# ============================================================
# RRF 融合
# ============================================================
def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]],
                           k: int | None = None) -> list[tuple[str, float]]:
    """
    RRF 融合多路召回结果。

    公式：score(d) = Σ_i  1 / (k + rank_i(d))
      · rank_i(d)：文档 d 在第 i 路里的名次（从 1 开始）
      · k：平滑常数（默认 60），越大越"平均"，越小越"赢者通吃"

    追问：如果某文档只在一路出现怎么办？
      → 另一路视作"未召回"，不加分。所以"两路都进前 10" 的文档
        会明显高于"只在单路第 1" 的文档 —— 这正是融合想要的效果：
        奖励**多路共识**，而不是单路的极端高分。
    """
    k = k or settings.rrf_k
    fused: dict[str, float] = defaultdict(float)
    hit_count: dict[str, int] = defaultdict(int)
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] += 1.0 / (k + rank)
            hit_count[cid] += 1
    # 次要排序键用命中路数，保证"两路都中"优先于"单路同分"
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], -hit_count[kv[0]]))
    return [(cid, round(score, 6)) for cid, score in ordered]


def _null_span():
    """无 trace 时的空上下文，避免到处写 if。"""
    from contextlib import nullcontext
    return nullcontext(_DummySpan())


class _DummySpan:
    model = None
    prompt_tokens = completion_tokens = 0


# ============================================================
# 混合检索器
# ============================================================
class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Index):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.meta = get_meta_store()

    def retrieve(self, query: str, top_k: int | None = None,
                 filters: dict | None = None,
                 trace: TraceRecorder | None = None) -> list[RetrievedChunk]:
        """
        完整检索流程：改写 → (HyDE) → 双路召回 → RRF 融合。
        filters 形如 {"dept": "技术部", "effective_before": 1730000000}，
        会被下推成存储层的过滤表达式（而不是召回后再筛）。
        """
        top_k = top_k or settings.retrieval_top_k
        expr = _build_expr(filters)

        # ---------- 1) 查询改写 ----------
        with (trace.span("retrieve", kind="retrieve") if trace else _null_span()):
            rewritten = rewrite_query(query, trace=trace)

            # ---------- 2) 双路召回 ----------
            vec_ranking: list[tuple[str, float]] = []
            with (trace.span("vector_recall", kind="retrieve") if trace else _null_span()) as sp:
                queries_for_vec = [rewritten]
                hyde = hyde_expand(query, trace=trace)
                if hyde:
                    queries_for_vec.append(hyde)
                # 多查询向量召回：取并集的最好名次
                merged: dict[str, float] = {}
                for q in queries_for_vec:
                    emb = llm.embed([q], trace=trace, trace_span=sp)[0]
                    for cid, score in self.vector_store.search(emb, top_k, expr=expr):
                        merged[cid] = max(merged.get(cid, -1e9), score)
                vec_ranking = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]

            with (trace.span("bm25_recall", kind="retrieve") if trace else _null_span()):
                bm25_ranking = self.bm25.search(rewritten, top_k)

            # ---------- 3) RRF 融合 ----------
            with (trace.span("rrf_fusion", kind="retrieve") if trace else _null_span()):
                fused = reciprocal_rank_fusion([vec_ranking, bm25_ranking])[: settings.fusion_top_k]

        # ---------- 4) 回填文本与元数据 ----------
        pos_v = {cid: i for i, (cid, _) in enumerate(vec_ranking, 1)}
        pos_b = {cid: i for i, (cid, _) in enumerate(bm25_ranking, 1)}
        meta_map = self.meta.get_chunk_meta([cid for cid, _ in fused])
        texts = self._get_texts([cid for cid, _ in fused])

        out: list[RetrievedChunk] = []
        for cid, score in fused:
            if cid not in texts:
                continue
            m = meta_map.get(cid, {})
            out.append(RetrievedChunk(
                chunk_id=cid, text=texts[cid], score=score, source="hybrid",
                parent_id=m.get("parent_id"), meta=m,
                rank_vector=pos_v.get(cid), rank_bm25=pos_b.get(cid),
            ))
        logger.info("检索完成：改写=%r 向量=%d BM25=%d 融合=%d",
                    rewritten[:40], len(vec_ranking), len(bm25_ranking), len(out))
        return out

    def _get_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """从向量库元数据或 SQL 取原文。这里优先取 child 文本。"""
        if not chunk_ids:
            return {}
        # 本项目的文本同时写入 meta_store.chunks，直接从 SQL 取（比从向量库取稳）
        rows = self.meta._run(
            f"SELECT chunk_id,text FROM chunks WHERE chunk_id IN ({','.join([self.meta._ph]*len(chunk_ids))})",
            tuple(chunk_ids), fetch=True)
        got = {r["chunk_id"]: r["text"] for r in rows}
        # 兜底：本地向量库场景下文本可能只在 meta jsonl
        missing = [c for c in chunk_ids if c not in got]
        if missing:
            for m in getattr(self.vector_store, "_meta", []) or []:
                if m.get("chunk_id") in missing:
                    got[m["chunk_id"]] = m.get("text", "")
        return got


def _build_expr(filters: dict | None) -> str | None:
    """
    把 filters 翻译成 Milvus 过滤表达式。
    设计要点：为什么在存储层过滤？
      → 下推（push-down）能在 ANN 搜索时就排除掉无权文档，
        不会把召回预算浪费在不可见的块上。召回后过滤会导致"看似召回 10 条、
        筛完 0 条"的假阴性。
    """
    if not filters:
        return None
    parts = []
    if filters.get("dept"):
        parts.append(f'dept == "{filters["dept"]}"')
    if filters.get("doc_type"):
        parts.append(f'doc_type == "{filters["doc_type"]}"')
    if filters.get("effective_before"):
        parts.append(f"effective_ts <= {int(filters['effective_before'])}")
    if filters.get("effective_after"):
        parts.append(f"effective_ts >= {int(filters['effective_after'])}")
    return " and ".join(parts) if parts else None


def build_bm25_from_store() -> BM25Index:
    """从已落库的 chunks 重建 BM25（服务启动时调用一次）。"""
    meta = get_meta_store()
    rows = meta._run("SELECT chunk_id,text FROM chunks ORDER BY chunk_id", fetch=True)
    chunks = [Chunk(chunk_id=r["chunk_id"], doc_id="", text=r["text"] or "") for r in rows]
    idx = BM25Index()
    idx.build(chunks)
    return idx
