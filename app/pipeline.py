"""
知枢 · 主流程编排
============================================================
把各层串成两条主链路：
  · 入库链路：解析 → 分块 → 向量化 → 写 Milvus + PG
  · 问答链路：改写 → 双路召回 → RRF → 重排 → 生成治理 → 溯源
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.llm_client import llm
from app.core.observability import TraceRecorder, metrics
from app.generation.rerank_generate import generate_answer, reranker
from app.ingest.chunker import chunk_document
from app.ingest.parsers import ParsedDoc, parse_document
from app.retrieval.hybrid_retriever import BM25Index, HybridRetriever
from app.storage.cache import bump_kb_version, get_cache
from app.storage.meta_store import get_meta_store
from app.storage.vector_store import Chunk as VecChunk, build_vector_store

logger = logging.getLogger("zhishu.pipeline")


class ZhishuPipeline:
    """
    知识中枢的门面（Facade）。对外只暴露 ingest / ask / stats 三个方法，
    内部各层可独立替换与测试 —— 这是"高内聚低耦合"的落地。
    """

    def __init__(self) -> None:
        self.vector_store = build_vector_store()
        self.meta = get_meta_store()
        self.cache = get_cache()
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(self.vector_store, self.bm25)
        self._bm25_ready = False
        logger.info("知枢 Pipeline 初始化完成 | 向量=%s | 元数据=%s",
                    type(self.vector_store).__name__, self.meta.backend)

    # ============================================================
    # 入库链路
    # ============================================================
    def ingest(self, path: str | Path, dept: str = "", effective_ts: int = 0,
               title: str | None = None) -> dict:
        """
        单文档入库。幂等：同一文件重复入库会覆盖而非重复追加（doc_id 基于路径）。
        """
        path = Path(path)
        trace = TraceRecorder(task_id=f"ingest_{path.stem}")
        t0 = time.perf_counter()
        try:
            with trace.span("parse", kind="ingest", file=path.name):
                doc = parse_document(path)
            doc.dept = dept  # type: ignore[attr-defined]
            doc.effective_ts = effective_ts  # type: ignore[attr-defined]
            if title:
                doc.title = title

            with trace.span("chunk", kind="chunk"):
                children, parents = chunk_document(doc)

            with trace.span("embed", kind="embed"):
                # -------- 只对 child 建向量（父块不参与向量检索）--------
                # 面试考点：为什么父块不建向量？
                #   → 父块是"生成上下文"，不是"检索单元"。若父块也建向量，
                #     它会和子块在同一空间竞争，且大块向量语义被稀释，
                #     反而拉低召回精度。父子块的正确用法是：
                #     **只检索子块 → 命中后回溯父块喂 LLM**。
                texts = [c.text for c in children]
                vecs = llm.embed(texts, trace=trace) if texts else []
                for c, v in zip(children, vecs):
                    c.meta["_vec"] = v

            with trace.span("persist", kind="store"):
                # 1) 元数据（PG / SQLite）
                self.meta.upsert_document({
                    "doc_id": doc.doc_id, "title": doc.title, "source_path": str(path),
                    "doc_type": _doc_type(doc), "dept": dept,
                    "effective_ts": effective_ts, "page_count": doc.page_count,
                })
                self.meta.insert_chunks([
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "parent_id": c.parent_id,
                     "text": c.text, "ordinal": c.ordinal, "token_est": c.token_est}
                    for c in children + parents])
                # 2) 向量（Milvus / 本地）—— 只入 child
                self._drop_doc_vectors(doc.doc_id)
                self.vector_store.insert([
                    VecChunk(chunk_id=c.chunk_id, doc_id=c.doc_id, parent_id=c.parent_id,
                             text=c.text, embedding=c.meta.pop("_vec"), meta=c.meta)
                    for c in children if "_vec" in c.meta])

            # 重建 BM25 并让旧缓存失效
            self.refresh_index()
            bump_kb_version()

            trace.save()
            return {
                "ok": True, "doc_id": doc.doc_id, "title": doc.title,
                "parser": doc.parser, "pages": doc.page_count,
                "children": len(children), "parents": len(parents),
                "warnings": doc.warnings,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "trace_id": trace.trace_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("入库失败: %s", path)
            trace.save()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace_id": trace.trace_id}

    def _drop_doc_vectors(self, doc_id: str) -> None:
        """
        幂等：同一文档重新入库时先清掉旧向量。
        面试坑：Milvus 的 delete 要求 collection 已 load，否则报
        "collection not loaded"。新手常在这里栽跟头 —— 所以先 load 再 delete。
        """
        try:
            if hasattr(self.vector_store, "collection"):
                self.vector_store.collection.load()
                self.vector_store.collection.delete(f'doc_id == "{doc_id}"')
                self.vector_store.collection.flush()
            else:
                store = self.vector_store
                if getattr(store, "_meta", None):
                    keep = [(v, m) for v, m in zip(store._vectors, store._meta)
                            if m.get("doc_id") != doc_id]
                    import numpy as np
                    store._vectors = np.asarray([v for v, _ in keep]) if keep else None
                    store._meta = [m for _, m in keep]
                    store._persist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理旧向量失败（忽略）: %s", exc)

    def refresh_index(self) -> None:
        self.bm25 = BM25Index()
        rows = self.meta._run("SELECT chunk_id,text FROM chunks ORDER BY chunk_id", fetch=True)
        from app.ingest.chunker import Chunk
        chunks = [Chunk(chunk_id=r["chunk_id"], doc_id="", text=r["text"] or "") for r in rows]
        self.bm25.build(chunks)
        self.retriever.bm25 = self.bm25
        self._bm25_ready = True

    # ============================================================
    # 问答链路
    # ============================================================
    def ask(self, query: str, filters: dict | None = None,
            use_cache: bool = True) -> dict:
        """一次完整问答。返回答案 + 引用 + 溯源信息 + trace。"""
        if not self._bm25_ready:
            self.refresh_index()

        # ---------- 缓存：完全相同的提问直接命中 ----------
        cache_key = f"{query}|{filters}"
        if use_cache:
            hit = self.cache.get("answer", cache_key)
            if hit is not None:
                hit["cached"] = True
                return hit

        trace = TraceRecorder(task_id=f"ask_{uuid.uuid4().hex[:8]}", query=query)
        refused = False
        try:
            candidates = self.retriever.retrieve(query, filters=filters, trace=trace)
            # 重排内部会做 small-to-big 父块回溯（命中 child → 喂 parent），
            # 并同步登记证据原文（供引用校验），所以这里无需再手动注册。
            ranked = reranker.rerank(query, candidates, trace=trace)
            gen = generate_answer(query, ranked, trace=trace)
            refused = bool(gen.get("refused"))

            result = {
                "question": query,
                "answer": gen.get("answer", ""),
                "citations": gen.get("citations", []),
                "confidence": gen.get("confidence", 0.0),
                "refused": refused,
                "reason": gen.get("reason", ""),
                "sources": [
                    {"evidence_id": i, "chunk_id": r.chunk_id, "score": r.score,
                     "text": r.text[:300]}
                    for i, r in enumerate(ranked, start=1)
                ],
                "retrieval_stats": {
                    "retrieved": len(candidates),
                    "reranked": len(ranked),
                    "top_score": gen.get("top_score", 0.0),
                },
                "latency_ms": trace.total_latency_ms,
                "stage_breakdown_ms": trace.stage_breakdown(),
                "total_tokens": trace.total_tokens,
                "cost_usd": trace.total_cost_usd,
                "trace_id": trace.trace_id,
                "cached": False,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("问答失败")
            result = {"question": query, "answer": "", "citations": [], "refused": True,
                      "reason": f"系统异常：{type(exc).__name__}: {exc}",
                      "sources": [], "trace_id": trace.trace_id, "cached": False}

        trace.save()
        metrics.record_query(trace, ok=not result.get("reason", "").startswith("系统异常"),
                             refused=refused)
        self.meta.log_qa({"id": trace.trace_id, "question": query,
                          "answer": result.get("answer", ""),
                          "citations": result.get("citations", []),
                          "refused": refused, "latency_ms": trace.total_latency_ms})
        if use_cache and not refused and result.get("answer"):
            self.cache.set("answer", cache_key, result)
        return result

    # ============================================================
    def stats(self) -> dict:
        return {
            "vector_backend": type(self.vector_store).__name__,
            "vector_count": self.vector_store.count(),
            "metadata_backend": self.meta.backend,
            "cache_backend": type(self.cache).__name__,
            "db": self.meta.stats(),
            "metrics": metrics.snapshot(),
            "llm_mock": llm.mock,
        }


def _doc_type(doc: ParsedDoc) -> str:
    return getattr(doc, "_doc_type", "general")


_pipeline: ZhishuPipeline | None = None


def get_pipeline() -> ZhishuPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ZhishuPipeline()
    return _pipeline
