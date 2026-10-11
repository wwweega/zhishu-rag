"""
知枢 · 企业级 RAG 知识中枢
============================
配置中心：所有可变参数集中在这里，通过环境变量 / .env 覆盖。

设计原则（面试考点）：
  1. 配置与代码分离 —— 换模型、换库、换阈值不用改业务代码。
  2. 每个外部依赖都要有"降级路径" —— Milvus/PG/Redis 任一不可用时，
     系统必须能启动并明确告知降级到什么模式，而不是直接崩。
     这对应生产环境的现实：本地开发往往只有一半的中间件。
  3. 不写死密钥 —— 只从环境读取，.env 进 .gitignore。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 应用 ----------------
    app_name: str = "知枢 Zhishu · 企业级 RAG 知识中枢"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"

    # ---------------- LLM（OpenAI 兼容：百炼 / DeepSeek / OpenAI）----------------
    llm_provider: str = "openai_compatible"  # openai_compatible | mock
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_small_model: str = "qwen-turbo"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # ---------------- Embedding ----------------
    # 说明：embedding 走百炼 text-embedding-v3（1024 维），本地无需跑模型。
    embed_provider: str = "openai_compatible"  # openai_compatible | hash(离线兜底)
    embed_model: str = "text-embedding-v3"
    embed_dim: int = 1024
    embed_batch_size: int = 16

    # ---------------- Rerank ----------------
    # bge-reranker-v2-m3，百炼提供 OpenAI 不兼容的原生 rerank 接口。
    rerank_provider: str = "dashscope"  # dashscope | mock
    rerank_model: str = "gte-rerank-v2"
    rerank_top_n: int = 6

    # ---------------- 存储：Milvus ----------------
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "zhishu_chunks"
    # 降级：Milvus 不可用时用内置的 numpy 暴力检索（小规模够用）
    vector_backend: str = "milvus"  # milvus | local

    # ---------------- 存储：PostgreSQL ----------------
    database_url: str = "postgresql://zhishu:zhishu@localhost:5432/zhishu"
    # 降级：PG 不可用时用 SQLite（元数据 / 会话 / 评测记录）
    metadata_backend: str = "postgres"  # postgres | sqlite

    # ---------------- 缓存：Redis ----------------
    redis_url: str = "redis://localhost:6379/0"
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    # 降级：Redis 不可用时用进程内 LRU
    cache_backend: str = "redis"  # redis | memory

    # ---------------- 检索参数 ----------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20        # 双路各召回多少
    fusion_top_k: int = 10           # RRF 融合后保留多少
    rrf_k: int = 60                  # RRF 公式常数
    use_query_rewrite: bool = True
    use_hyde: bool = True

    # ---------------- 生成治理 ----------------
    refuse_threshold: float = 0.35   # 证据不足拒答阈值（归一化后，见 refuse_norm_threshold）
    refuse_norm_threshold: float = 0.25  # 批内归一化 top 分低于此值 → 视为"整批不相关"，拒答
    refuse_floor: float = 0.0            # 绝对分兜底：仅当归一化低 **且** 绝对分≈0 才拒答
    citation_required: bool = True

    # ---------------- 可观测 ----------------
    enable_trace: bool = True
    trace_dir: str = str(ROOT / "data" / "traces")
    enable_cost_tracking: bool = True

    # ---------------- 路径 ----------------
    corpus_dir: str = str(ROOT / "data" / "corpus")
    index_dir: str = str(ROOT / "data" / "index")
    reports_dir: str = str(ROOT / "data" / "reports")

    @property
    def trace_path(self) -> Path:
        p = Path(self.trace_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def llm_is_mock(self) -> bool:
        return self.llm_provider == "mock" or not self.llm_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
