"""
知枢 · 应用入口
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO),
                    format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s")
logger = logging.getLogger("zhishu.main")

app = FastAPI(
    title=settings.app_name,
    description="企业私有文档的 RAG 问答中枢：混合检索 + 引用溯源 + 拒答 + 全链路可观测",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(FRONTEND / "index.html"))


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.on_event("startup")
def _warmup() -> None:
    """
    启动预热：构建 BM25 索引。
    设计要点：为什么预热而不懒加载？
      → 第一个用户的请求不应该承担"建索引"的开销（否则 P99 毛刺极大）。
        预热是长链路服务的基本功。
    """
    try:
        from app.pipeline import get_pipeline
        p = get_pipeline()
        p.refresh_index()
        logger.info("预热完成 | 向量后端=%s | 元数据=%s | 缓存=%s",
                    type(p.vector_store).__name__, p.meta.backend, type(p.cache).__name__)
    except Exception as exc:  # noqa: BLE001
        logger.warning("预热失败（服务仍可启动，首次请求会补建索引）: %s", exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
