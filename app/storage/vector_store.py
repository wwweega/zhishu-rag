"""
知枢 · 向量存储层（Milvus，带本地降级）
============================================================
面试考点：Milvus / Qdrant / pgvector 怎么选？

  · Milvus：分布式、十亿级、支持多种索引（HNSW/IVF/DiskANN），
            适合"独立扩容、数据量大、要独立运维"的场景。本项目选它
            是为了练分布式向量库 + 独立扩容的工程能力。
  · pgvector：向量和业务数据同库，少一个中间件，中小规模最省心。
  · Qdrant：Rust 写的，单机性能好，过滤能力强。

  取舍：知枢选 Milvus 做向量主力，PostgreSQL 只存元数据与权限 ——
        把"重检索"和"重事务"分开，各自选型。
        如果规模在百万以内，我会反过来直接用 pgvector 少运维一套。

设计要点（面试会问）：
  1. collection schema：主键 / 向量 / 标量字段（用于元数据过滤）
  2. 索引：HNSW（M=16, efConstruction=200）—— 召回率与内存的平衡点
  3. 一致性：Milvus 默认 eventual；插入后要 flush 才能被检索到
  4. 降级：Milvus 不可用时切 LocalVectorStore（numpy 暴力检索）
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import settings

logger = logging.getLogger("zhishu.vector")


class Chunk:
    """一条入库的最小单元。父子块结构：child 用于检索，parent 用于喂给 LLM。"""

    __slots__ = ("chunk_id", "doc_id", "parent_id", "text", "embedding", "meta")

    def __init__(self, chunk_id: str, doc_id: str, parent_id: str | None,
                 text: str, embedding: list[float] | None = None, meta: dict | None = None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.parent_id = parent_id
        self.text = text
        self.embedding = embedding
        self.meta = meta or {}


class VectorStore:
    """
    统一接口：insert / search / count / drop。
    具体后端由 settings.vector_backend 决定，业务层无感知。
    """

    def insert(self, chunks: Sequence[Chunk]) -> int:
        raise NotImplementedError

    def search(self, query_vec: list[float], top_k: int,
               expr: str | None = None) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，score 越大越相关。"""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def drop(self) -> None:
        raise NotImplementedError


# ============================================================
# Milvus 后端
# ============================================================
class MilvusVectorStore(VectorStore):
    def __init__(self, host: str | None = None, port: int | None = None,
                 collection: str | None = None):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        self._utility = utility
        self.collection_name = collection or settings.milvus_collection
        connections.connect(alias="default", host=host or settings.milvus_host,
                            port=str(port or settings.milvus_port), timeout=10)
        self._ensure_collection()
        self.collection = Collection(self.collection_name)

    def _ensure_collection(self) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if utility.has_collection(self.collection_name):
            return
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="dept", dtype=DataType.VARCHAR, max_length=64),      # 部门过滤
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="effective_ts", dtype=DataType.INT64),               # 时效过滤
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embed_dim),
        ]
        schema = CollectionSchema(fields, description="知枢知识块（child chunk）")
        col = Collection(self.collection_name, schema)
        # HNSW：M 控制图连接度（越大召回越好、内存越大）；efConstruction 建索引时的候选集
        col.create_index("embedding", {
            "index_type": "HNSW", "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200},
        })
        logger.info("已创建 Milvus collection=%s", self.collection_name)

    def insert(self, chunks: Sequence[Chunk]) -> int:
        rows = [[], [], [], [], [], [], [], []]
        for c in chunks:
            if c.embedding is None:
                continue
            rows[0].append(c.chunk_id)
            rows[1].append(c.doc_id)
            rows[2].append(c.parent_id or "")
            rows[3].append(c.text[:8000])
            rows[4].append(str(c.meta.get("dept", ""))[:60])
            rows[5].append(str(c.meta.get("doc_type", ""))[:30])
            rows[6].append(int(c.meta.get("effective_ts", 0)))
            rows[7].append(c.embedding)
        if not rows[0]:
            return 0
        self.collection.insert(rows)
        # 面试点：Milvus 是"写后可见性最终一致"，flush 后才能保证被检索到
        self.collection.flush()
        return len(rows[0])

    def search(self, query_vec: list[float], top_k: int, expr: str | None = None
               ) -> list[tuple[str, float]]:
        self.collection.load()
        res = self.collection.search(
            data=[query_vec], anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 128}},  # ef 越大越准越慢
            limit=top_k, expr=expr, output_fields=["chunk_id"],
        )
        out = []
        for hits in res:
            for h in hits:
                out.append((h.entity.get("chunk_id"), round(float(h.score), 4)))
        return out

    def count(self) -> int:
        self.collection.load()
        return self.collection.num_entities

    def drop(self) -> None:
        """
        清空向量数据。
        面试坑：Milvus 删除 collection 后，本对象持有的 self.collection 句柄就失效了，
        再 insert 会报 "collection not found"。正确做法是 **drop 后立刻重建**，
        并把 self.collection 重新指向新 collection —— 这样 drop() 可以安全地
        被"清库重导"流程反复调用，符合大家对 clear()/drop() 的直觉。
        """
        from pymilvus import Collection

        if self._utility.has_collection(self.collection_name):
            self._utility.drop_collection(self.collection_name)
        self._ensure_collection()
        self.collection = Collection(self.collection_name)
        logger.info("已清空并重建 Milvus collection=%s", self.collection_name)


# ============================================================
# 本地降级后端（numpy 暴力检索）
# ============================================================
class LocalVectorStore(VectorStore):
    """
    无 Milvus 时用。把向量存成 .npy + 元数据 jsonl，检索时全量点积。
    面试考点：什么时候可以暴力检索？
      → 数据量 < 10 万且 QPS 不高时，暴力检索（brute force）完全够用，
        而且零运维。HNSW 的近似召回在低延迟高并发下才必要。
        "先量一量规模再选型"比"一上来就上分布式"更专业。
    """

    def __init__(self, path: str | None = None):
        self.dir = Path(path or settings.index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_file = self.dir / "vectors.npy"
        self.meta_file = self.dir / "chunks.jsonl"
        self._vectors: np.ndarray | None = None
        self._meta: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.vec_file.exists() and self.meta_file.exists():
            self._vectors = np.load(self.vec_file)
            self._meta = [json.loads(line) for line in
                          self.meta_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _persist(self) -> None:
        if self._vectors is not None:
            np.save(self.vec_file, self._vectors)
        with self.meta_file.open("w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def insert(self, chunks: Sequence[Chunk]) -> int:
        new_vecs, new_meta = [], []
        for c in chunks:
            if c.embedding is None:
                continue
            new_vecs.append(c.embedding)
            new_meta.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                             "parent_id": c.parent_id, "text": c.text, "meta": c.meta})
        if not new_vecs:
            return 0
        arr = np.asarray(new_vecs, dtype=np.float32)
        self._vectors = arr if self._vectors is None else np.vstack([self._vectors, arr])
        self._meta.extend(new_meta)
        self._persist()
        return len(new_vecs)

    def search(self, query_vec: list[float], top_k: int, expr: str | None = None
               ) -> list[tuple[str, float]]:
        if self._vectors is None or len(self._meta) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        # 归一化后点积 == 余弦相似度
        qn = q / (np.linalg.norm(q) or 1.0)
        vn = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9)
        scores = vn @ qn
        order = np.argsort(-scores)[: top_k * 3]
        out = []
        for idx in order:
            m = self._meta[int(idx)]
            if expr and not _match_expr(m, expr):
                continue
            out.append((m["chunk_id"], round(float(scores[int(idx)]), 4)))
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        return len(self._meta)

    def drop(self) -> None:
        self._vectors, self._meta = None, []
        for f in (self.vec_file, self.meta_file):
            f.unlink(missing_ok=True)


def _match_expr(meta: dict, expr: str) -> bool:
    """
    极简 expr 过滤（仅本地后端用）：支持 'dept == "技术部"' / 'effective_ts >= 123'。
    生产应交给 Milvus 的标量过滤（下推执行，不消耗召回预算）。
    """
    try:
        key = expr.split()[0]
        op = expr.split()[1]
        val = expr.split('"')[1] if '"' in expr else expr.split()[2]
        cur = meta.get("meta", {}).get(key, meta.get(key))
        if op == "==":
            return str(cur) == val
        if op == ">=":
            return float(cur or 0) >= float(val)
        if op == "<=":
            return float(cur or 0) <= float(val)
    except Exception:  # noqa: BLE001
        return True
    return True


# ============================================================
# 工厂：自动选择后端 + 降级
# ============================================================
def build_vector_store() -> VectorStore:
    """
    面试考点：为什么在这里 try/except 而不是让上层处理？
      → 中间件可用性是部署环境问题，不该污染业务逻辑。
        在"工厂"这一层把降级决策做完，业务层永远只看到 VectorStore 接口。
        同时必须**大声打日志**，否则线上会静默跑在降级模式而不自知。
    """
    if settings.vector_backend == "local":
        logger.warning("vector_backend=local，使用本地暴力检索（非 Milvus）")
        return LocalVectorStore()
    try:
        store = MilvusVectorStore()
        logger.info("向量后端：Milvus @ %s:%s", settings.milvus_host, settings.milvus_port)
        return store
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus 不可用（%s），降级到本地暴力检索", exc)
        return LocalVectorStore()
