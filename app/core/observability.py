"""
知枢 · 可观测与成本追踪
============================================================
面试考点：RAG 系统为什么必须做全链路观测？

  一个 RAG 问答"慢"，可能是四段里任意一段：
    ① 解析（OCR / 表格还原）  ② 召回（向量 + BM25）  ③ 重排（reranker）  ④ 生成（LLM）
  没有分段 Trace，你只能看到总数 12s，然后开始瞎猜。
  有了分段 Trace，你会发现 11.2s 花在 rerank 上 —— 这才是可优化的。

  另外，RAG 的成本大头在「喂给 LLM 的 context token」，
  所以必须记录每次生成的 prompt_tokens，否则"降本"无从谈起。

本模块复用与「运营智脑」同一套 trace_id / parent_span_id 父子模型，
因为这就是分布式追踪（OpenTelemetry）的基本模型，一通百通。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from app.core.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zhishu")


# ============================================================
# 价格表（USD / 1K tokens）。价格会变 —— 这里是配置不是硬编码业务逻辑。
# 百炼报价单位是「元/百万 token」，按汇率 7.2 换算成 USD/1K。
# ============================================================
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model: (prompt_per_1k, completion_per_1k)
    "qwen-plus": (0.0004, 0.0012),
    "qwen-turbo": (0.00005, 0.0002),
    "qwen-max": (0.001667, 0.005),
    "text-embedding-v3": (0.00007, 0.0),   # embedding 按输入计费，约 0.5 元/M
    "gte-rerank-v2": (0.00011, 0.0),
    "deepseek-chat": (0.00027, 0.0011),
    "mock": (0.0, 0.0),
}


def lookup_price(model: str) -> tuple[float, float]:
    """查单价，支持前缀匹配兜底（模型常带日期快照后缀，如 qwen-plus-0915）。"""
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    for name, price in PRICE_TABLE.items():
        if name != "mock" and model.startswith(name):
            return price
    return (0.0, 0.0)


@dataclass
class Span:
    """一次被追踪的调用：解析 / 分块 / 召回 / 重排 / 生成 / LLM 请求。"""

    span_id: str
    name: str
    kind: str  # ingest | chunk | retrieve | rerank | generate | llm | embed | store
    parent_id: str | None = None
    start_ts: float = field(default_factory=time.perf_counter)
    end_ts: float | None = None
    status: str = "running"  # running | ok | error
    error: str | None = None
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        end = self.end_ts if self.end_ts is not None else time.perf_counter()
        return round((end - self.start_ts) * 1000, 2)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = self.latency_ms
        d["total_tokens"] = self.total_tokens
        return d


class TraceRecorder:
    """
    一次问答 / 一次入库任务的完整调用树。

    面试追问：为什么手写而不直接上 Langfuse？
      → 生产我会用 Langfuse / OTel Collector；这里手写是为了把
        trace_id / parent_span_id 的父子关系讲透 —— 这正是模型服务
        里最基础的分布式追踪模型，也是"能定位慢在哪一步"的根。
        （本项目同时预留了 Langfuse 上报接口，见 export_to_langfuse。）
    """

    def __init__(self, task_id: str, query: str = "", persist: bool = True):
        self.task_id = task_id
        self.query = query
        self.persist = persist
        self.trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "node", **meta: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        sp = Span(
            span_id=f"sp_{uuid.uuid4().hex[:8]}",
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            meta=meta,
        )
        self.spans.append(sp)
        self._stack.append(sp)
        try:
            yield sp
            sp.status = "ok"
        except Exception as exc:  # noqa: BLE001
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            logger.exception("span 失败: %s", name)
            raise
        finally:
            sp.end_ts = time.perf_counter()
            self._stack.pop()

    def record_llm(self, span: Span, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        px_in, px_out = lookup_price(model)
        cost = prompt_tokens / 1000 * px_in + completion_tokens / 1000 * px_out
        span.model = model
        span.prompt_tokens += prompt_tokens
        span.completion_tokens += completion_tokens
        span.cost_usd = round(span.cost_usd + cost, 6)
        return cost

    # ---------------- 汇总 ----------------
    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def total_latency_ms(self) -> float:
        """
        真实墙钟耗时 = 所有 span 里「最晚结束」减「最早开始」。

        面试考点：为什么不是 `max(某个 span 的 latency)`？
          → RAG 的四个阶段是顺序执行的，最长的那一段绝不等于端到端。
            只有取时间区间并集才是"用户实际等了多久"。
            这个口径错误在「运营智脑」项目里真实出现过（偏差 3.7 倍），
            所以在知枢里从第一天就用正确实现。
        """
        if not self.spans:
            return 0.0
        now = time.perf_counter()
        starts = [s.start_ts for s in self.spans]
        ends = [s.end_ts if s.end_ts is not None else now for s in self.spans]
        return round((max(ends) - min(starts)) * 1000, 2)

    def stage_breakdown(self) -> dict[str, float]:
        """
        按阶段归并耗时 —— 面试时直接给出"慢在哪一段"。

        面试考点（这里藏着一个真实 bug）：
          早期实现是 `sum(每个同 kind span 的 latency)`。但一个阶段内部常常
          **嵌套了子 span**：例如 retrieve 阶段下挂着 query_rewrite / vector_recall
          / hyde / bm25_recall / rrf_fusion 五个子 span。直接求和会把子 span 的耗时
          重复计入，导致 "retrieve 阶段 10690ms" 甚至**超过端到端墙钟 5950ms** 这种
          物理上不可能的结果 —— 一眼假。
          正确做法：每个阶段取**其所有 span 时间区间的并集长度**（重叠只算一次），
          这才是该阶段真实占用的墙钟时间。与 total_latency_ms 同源同口径。
        """
        stages: dict[str, list[tuple[float, float]]] = {}
        now = time.perf_counter()
        for sp in self.spans:
            if sp.kind not in ("ingest", "retrieve", "rerank", "generate"):
                continue
            end = sp.end_ts if sp.end_ts is not None else now
            stages.setdefault(sp.kind, []).append((sp.start_ts, end))
        return {k: round(_union_len_ms(v), 2) for k, v in stages.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "query": self.query,
            "created_at": self.created_at,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "wall_latency_ms": self.total_latency_ms,
            "stage_breakdown_ms": self.stage_breakdown(),
            "spans": [s.to_dict() for s in self.spans],
        }

    def save(self) -> Path | None:
        if not self.persist or not settings.enable_trace:
            return None
        path = settings.trace_path / f"{self.trace_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [
            f"┌─ Trace {self.trace_id} | 任务 {self.task_id}",
            f"│  墙钟 {self.total_latency_ms:.0f}ms | token {self.total_tokens}"
            f" | 成本 ${self.total_cost_usd:.5f}",
        ]
        br = self.stage_breakdown()
        lines.append(f"│  分段: 解析 {br['ingest']:.0f}ms | 召回 {br['retrieve']:.0f}ms"
                     f" | 重排 {br['rerank']:.0f}ms | 生成 {br['generate']:.0f}ms")
        lines.append("│  " + "-" * 60)
        for sp in sorted(self.spans, key=lambda s: -s.latency_ms)[:14]:
            mark = "✗" if sp.status == "error" else " "
            icon = {"ingest": "📄", "chunk": "✂️", "retrieve": "🔍", "rerank": "🎯",
                    "generate": "🧠", "llm": "🧠", "embed": "🧬", "store": "💾"}.get(sp.kind, "▫️")
            lines.append(f"│ {mark}{icon} {sp.name:<26} {sp.latency_ms:>8.1f}ms"
                         f"  {sp.total_tokens:>6}tk  ${sp.cost_usd:.5f}")
        lines.append("└" + "-" * 61)
        return "\n".join(lines)


def _union_len_ms(intervals: list[tuple[float, float]]) -> float:
    """合并一组时间区间，返回并集总长度（毫秒）。重叠区间只计一次。

    用于 stage_breakdown：同一阶段内嵌套/重叠的子 span 不能被重复求和。
    """
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:                 # 不相交 → 结算上一段
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:                         # 相交/嵌套 → 合并
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total * 1000


class MetricsCollector:
    """进程级指标聚合（环形缓冲，防止长跑内存无限增长）。"""

    def __init__(self, max_samples: int = 500):
        self.max_samples = max_samples
        self.latencies: list[float] = []
        self.tokens: list[int] = []
        self.costs: list[float] = []
        self.total_queries = 0
        self.failed_queries = 0
        self.refused_queries = 0     # 因证据不足被拒答
        self.cache_hits = 0
        self.cache_misses = 0

    def record_query(self, trace: TraceRecorder, ok: bool = True, refused: bool = False) -> None:
        self.total_queries += 1
        if not ok:
            self.failed_queries += 1
        if refused:
            self.refused_queries += 1
        self.latencies.append(trace.total_latency_ms)
        self.tokens.append(trace.total_tokens)
        self.costs.append(trace.total_cost_usd)
        for buf in (self.latencies, self.tokens, self.costs):
            if len(buf) > self.max_samples:
                del buf[0: len(buf) - self.max_samples]

    @staticmethod
    def _pct(buf: list[float], p: float) -> float:
        """P95：排序后取第 ceil(p*n) 位。关注尾部延迟，比均值更能反映体验。"""
        if not buf:
            return 0.0
        s = sorted(buf)
        idx = min(len(s) - 1, max(0, int(-(-p * len(s) // 1)) - 1))
        return round(s[idx], 2)

    def snapshot(self) -> dict[str, Any]:
        tot = self.total_queries
        return {
            "total_queries": tot,
            "failed_queries": self.failed_queries,
            "refused_queries": self.refused_queries,
            "success_rate": round(1 - self.failed_queries / tot, 4) if tot else 1.0,
            "refuse_rate": round(self.refused_queries / tot, 4) if tot else 0.0,
            "latency_ms": {
                "p50": self._pct(self.latencies, 0.50),
                "p95": self._pct(self.latencies, 0.95),
                "p99": self._pct(self.latencies, 0.99),
            },
            "total_tokens": sum(self.tokens),
            "total_cost_usd": round(sum(self.costs), 6),
            "avg_cost_per_query": round(sum(self.costs) / len(self.costs), 6) if self.costs else 0.0,
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hits / (self.cache_hits + self.cache_misses), 4)
                if (self.cache_hits + self.cache_misses) else 0.0,
            },
        }


metrics = MetricsCollector()
