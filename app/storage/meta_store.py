"""
知枢 · 元数据库（PostgreSQL，带 SQLite 降级）
============================================================
职责划分（面试会问"为什么不用一个库"）：

  · Milvus  → 只干向量检索（重检索、要独立扩容）
  · PostgreSQL → 元数据 + 权限 + 会话 + 评测记录（重事务、要 ACID）
  · Redis   → 缓存（热数据、可丢）

  这叫"按访问模式（access pattern）选存储"，而不是"技术栈堆料"。
  如果全都塞进 PostgreSQL，向量检索的召回性能会成为瓶颈；
  如果全都塞进 Milvus，复杂关系查询和事务会很难受。

表设计：
  documents   —— 文档主表（含部门 / 密级 / 生效时间，用于权限与时效过滤）
  chunks      —— 块表（含 parent_id 实现父子块）
  qa_logs     —— 问答日志（评测与回归用）
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger("zhishu.meta")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    source_path   TEXT,
    doc_type      TEXT,
    dept          TEXT,
    security      TEXT DEFAULT 'internal',
    effective_ts  BIGINT DEFAULT 0,
    page_count    INT DEFAULT 0,
    created_at    BIGINT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    parent_id  TEXT,
    text       TEXT,
    ordinal    INT,
    token_est  INT
);
CREATE TABLE IF NOT EXISTS qa_logs (
    id         TEXT PRIMARY KEY,
    question   TEXT,
    answer     TEXT,
    citations  TEXT,
    refused    INT DEFAULT 0,
    latency_ms REAL DEFAULT 0,
    created_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
"""


class MetaStore:
    """统一元数据接口，后端 postgres | sqlite 由 settings 决定。"""

    def __init__(self) -> None:
        self.backend = settings.metadata_backend
        self._conn = None
        self._init()

    def _init(self) -> None:
        if self.backend == "postgres":
            try:
                import psycopg2  # noqa: F401
                url = settings.database_url.replace("postgresql+psycopg2://", "postgresql://")
                self._conn = psycopg2.connect(url, connect_timeout=5)
                self._conn.autocommit = True
                self._exec_script(SCHEMA)
                logger.info("元数据后端：PostgreSQL")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("PostgreSQL 不可用（%s），降级到 SQLite", exc)
                self.backend = "sqlite"

        db = Path(settings.index_dir) / "zhishu_meta.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db), check_same_thread=False)
        self._exec_script(SCHEMA)
        logger.info("元数据后端：SQLite @ %s", db)

    # ---------------- 适配两种驱动的占位符差异 ----------------
    @property
    def _ph(self) -> str:
        return "%s" if self.backend == "postgres" else "?"

    def _exec_script(self, script: str) -> None:
        cur = self._conn.cursor()
        for stmt in script.split(";"):
            if stmt.strip():
                cur.execute(stmt)

    def _run(self, sql: str, params: tuple = (), fetch: bool = False):
        ph = self._ph
        sql = sql.replace("?", ph)
        cur = self._conn.cursor()
        cur.execute(sql, params)
        if fetch:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return cur.rowcount

    # ---------------- 写入 ----------------
    def upsert_document(self, doc: dict) -> None:
        self._run(
            "INSERT INTO documents (doc_id,title,source_path,doc_type,dept,security,"
            "effective_ts,page_count,created_at) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (doc_id) DO UPDATE SET title=EXCLUDED.title, page_count=EXCLUDED.page_count",
            (doc["doc_id"], doc.get("title", ""), doc.get("source_path", ""),
             doc.get("doc_type", ""), doc.get("dept", ""), doc.get("security", "internal"),
             int(doc.get("effective_ts", 0)), int(doc.get("page_count", 0)), int(time.time())),
        )

    def insert_chunks(self, chunks: list[dict]) -> int:
        n = 0
        for c in chunks:
            self._run(
                "INSERT INTO chunks (chunk_id,doc_id,parent_id,text,ordinal,token_est) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT (chunk_id) DO NOTHING",
                (c["chunk_id"], c["doc_id"], c.get("parent_id"), c.get("text", ""),
                 int(c.get("ordinal", 0)), int(c.get("token_est", 0))),
            )
            n += 1
        return n

    def log_qa(self, rec: dict) -> None:
        self._run(
            "INSERT INTO qa_logs (id,question,answer,citations,refused,latency_ms,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (rec["id"], rec.get("question", ""), rec.get("answer", ""),
             json.dumps(rec.get("citations", []), ensure_ascii=False),
             1 if rec.get("refused") else 0, float(rec.get("latency_ms", 0)), int(time.time())),
        )

    # ---------------- 读取 ----------------
    def get_parent_chunks(self, parent_ids: list[str]) -> dict[str, str]:
        """
        父子块的"取父块"：检索命中 child 后，用它回带完整父块喂给 LLM。
        面试考点：为什么不直接检索父块？
          → 父块太大，向量会被"稀释"（一篇文章的向量不聚焦），召回差；
            小块召回准、大块上下文全 —— 所以用小检索、大生成。
        """
        if not parent_ids:
            return {}
        ph = ",".join([self._ph] * len(parent_ids))
        rows = self._run(f"SELECT chunk_id,text FROM chunks WHERE chunk_id IN ({ph})",
                         tuple(parent_ids), fetch=True)
        return {r["chunk_id"]: r["text"] for r in rows}

    def get_chunk_meta(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        ph = ",".join([self._ph] * len(chunk_ids))
        rows = self._run(
            f"SELECT c.chunk_id,c.doc_id,c.parent_id,d.title,d.dept,d.security,d.doc_type "
            f"FROM chunks c LEFT JOIN documents d ON c.doc_id=d.doc_id "
            f"WHERE c.chunk_id IN ({ph})", tuple(chunk_ids), fetch=True)
        return {r["chunk_id"]: r for r in rows}

    def stats(self) -> dict[str, int]:
        docs = self._run("SELECT COUNT(*) AS n FROM documents", fetch=True)
        chunks = self._run("SELECT COUNT(*) AS n FROM chunks", fetch=True)
        qa = self._run("SELECT COUNT(*) AS n FROM qa_logs", fetch=True)
        return {"documents": docs[0]["n"] if docs else 0,
                "chunks": chunks[0]["n"] if chunks else 0,
                "qa_logs": qa[0]["n"] if qa else 0}

    def clear(self) -> None:
        for t in ("chunks", "documents", "qa_logs"):
            self._run(f"DELETE FROM {t}")


_meta: MetaStore | None = None


def get_meta_store() -> MetaStore:
    global _meta
    if _meta is None:
        _meta = MetaStore()
    return _meta
