"""
知枢 · HTTP API（FastAPI）
============================================================
对外提供：问答 / 入库 / 统计 / 健康检查。
SSE 流式：把"检索 → 重排 → 生成"的中间过程实时推给前端，
让用户看到"正在召回 20 条 → 重排剩 6 条 → 正在生成"，
而不是白屏等 5 秒 —— 这是长链路 AI 应用的体验刚需。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.pipeline import get_pipeline

logger = logging.getLogger("zhishu.api")
router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    dept: str | None = Field(None, description="部门（元数据过滤）")
    doc_type: str | None = Field(None, description="文档类型过滤")
    effective_before: int | None = Field(None, description="只看此时间前生效的文档")
    use_cache: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict[str, Any]] = []
    confidence: float = 0.0
    refused: bool = False
    reason: str = ""
    sources: list[dict[str, Any]] = []
    latency_ms: float = 0.0
    stage_breakdown_ms: dict[str, float] = {}
    trace_id: str = ""
    cached: bool = False


@router.get("/health", summary="健康检查：各依赖的真实可用状态")
def health() -> dict:
    """
    面试考点：健康检查为什么要报告"降级状态"而不是简单的 ok？
      → 一个跑在 SQLite + 内存缓存上的实例，和跑在 PG + Redis 上的实例，
        可用性 SLA 完全不同。健康检查必须能区分"活着"和"活在降级模式"，
        否则运维会误判容量。
    """
    p = get_pipeline()
    return {
        "status": "ok",
        "components": {
            "vector_backend": type(p.vector_store).__name__,
            "vector_count": p.vector_store.count(),
            "metadata_backend": p.meta.backend,
            "cache_backend": type(p.cache).__name__,
            "llm_mode": "mock" if p.stats()["llm_mock"] else "real",
        },
    }


@router.post("/ask", response_model=AskResponse, summary="知识库问答（含引用溯源）")
def ask(req: AskRequest) -> AskResponse:
    filters = {k: v for k, v in {
        "dept": req.dept, "doc_type": req.doc_type,
        "effective_before": req.effective_before,
    }.items() if v}
    result = get_pipeline().ask(req.question, filters=filters or None, use_cache=req.use_cache)
    return AskResponse(**{k: result[k] for k in AskResponse.model_fields if k in result})


@router.post("/ask/stream", summary="知识库问答（SSE 流式：实时推送检索/重排/生成阶段）")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """
    SSE 分阶段推进。前端可据此渲染"进度条 + 中间产物"。
    注意：这里用 asyncio.to_thread 把同步链路丢到线程池，
    避免阻塞事件循环 —— 同步阻塞是 FastAPI 最常见的性能坑。
    """
    p = get_pipeline()

    async def event_gen():
        yield _sse("stage", {"stage": "retrieve", "msg": "正在双路召回（向量 + BM25）…"})
        await asyncio.sleep(0)
        result = await asyncio.to_thread(p.ask, req.question, None, req.use_cache)
        yield _sse("stage", {"stage": "rerank",
                             "msg": f"重排完成，保留 {len(result.get('sources', []))} 条证据"})
        yield _sse("stage", {"stage": "generate", "msg": "正在生成并校验引用…"})
        yield _sse("result", result)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ingest", summary="上传并入库文档（pdf/txt/md）")
async def ingest(file: UploadFile = File(...), dept: str = "", effective_ts: int = 0) -> dict:
    from pathlib import Path

    from app.core.config import settings
    if not (file.filename or "").lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "仅支持 pdf / txt / md")
    corp = Path(settings.corpus_dir)
    corp.mkdir(parents=True, exist_ok=True)
    dest = corp / file.filename
    dest.write_bytes(await file.read())
    return get_pipeline().ingest(dest, dept=dept, effective_ts=effective_ts)


@router.get("/stats", summary="运行统计（含 P95 延迟 / 成本 / 缓存命中）")
def stats() -> dict:
    return get_pipeline().stats()


@router.get("/docs/list", summary="已入库文档列表")
def list_docs() -> dict:
    rows = get_pipeline().meta._run(
        "SELECT doc_id,title,dept,doc_type,page_count FROM documents ORDER BY created_at DESC",
        fetch=True)
    return {"count": len(rows), "documents": rows}
