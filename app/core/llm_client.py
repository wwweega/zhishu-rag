"""
知枢 · LLM / Embedding / Rerank 统一客户端
============================================================
设计要点：为什么要包一层 client，而不直接在业务里调 openai？

  1. 换供应商不动业务代码 —— 百炼 / DeepSeek / OpenAI 都是 OpenAI 兼容协议，
     差异只在 base_url + model 名，抽象掉之后业务层零感知。
  2. 统一重试与降级 —— 网络抖动、限流（429）、超时，在这里统一处理。
       · 瞬时错误（429 / 5xx / 超时）→ 指数退避重试
       · 确定性错误（400 / 参数非法）→ 不重试，直接暴露，重试只会白烧钱
     这个"错误分类"是生产系统与 demo 的分界线。
  3. 没配 Key 时自动 mock —— 保证 clone 下来的人不填 Key 也能跑通全链路，
     这是开源项目的礼貌，也是 CI 能在无密钥环境跑测试的前提。
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.core.observability import Span, TraceRecorder

logger = logging.getLogger("zhishu.llm")


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    mocked: bool = False


class TransientError(Exception):
    """瞬时错误：值得重试（限流 / 5xx / 超时）。"""


class DeterministicError(Exception):
    """确定性错误：重试无用（400 / 鉴权失败 / 参数错）。"""


# ============================================================
# 错误分类：决定"要不要重试"的唯一依据
# ============================================================
def classify_http_error(status: int, body: str = "") -> type[Exception]:
    """
    设计要点：429 和 400 的区别？
      → 429 Too Many Requests 是"你太快了"，退避后重试通常能成功。
      → 400 Bad Request 是"你请求写错了"，重试 100 次也一样错，只会烧钱。
      把它们混为一谈（无脑重试）是新手最常见的生产事故来源。
    """
    if status in (408, 409, 425, 429) or status >= 500:
        return TransientError
    return DeterministicError


def _backoff(attempt: int, base: float = 0.6, cap: float = 8.0) -> float:
    """指数退避 + 抖动。抖动是为了避免多个副本同时重试造成"惊群"。"""
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random() * 0.5)


class LLMClient:
    """OpenAI 兼容协议客户端，覆盖 chat / embedding / rerank。"""

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.mock = settings.llm_is_mock

    # ---------------- 底层 POST ----------------
    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_exc: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            try:
                resp = httpx.post(url, json=payload, headers=headers,
                                  timeout=timeout or settings.llm_timeout)
                if resp.status_code == 200:
                    return resp.json()
                err_cls = classify_http_error(resp.status_code, resp.text[:300])
                msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if err_cls is DeterministicError:
                    raise DeterministicError(msg)
                last_exc = TransientError(msg)
                logger.warning("瞬时错误 retry=%d: %s", attempt + 1, msg)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = TransientError(f"{type(exc).__name__}: {exc}")
                logger.warning("网络错误 retry=%d: %s", attempt + 1, exc)
            time.sleep(_backoff(attempt))
        raise last_exc or TransientError("未知失败")

    # ---------------- Chat ----------------
    def chat(self, prompt: str, system: str = "", model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool = False, trace: TraceRecorder | None = None,
             trace_span: Span | None = None) -> ChatResult:
        model = model or settings.llm_model
        if self.mock:
            res = self._mock_chat(prompt, json_mode)
            if trace and trace_span:
                trace.record_llm(trace_span, "mock", res.prompt_tokens, res.completion_tokens)
            return res

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        # 结构化输出：百炼/OpenAI 兼容接口都支持 response_format
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = self._post("/chat/completions", payload)
        usage = data.get("usage", {})
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        text = data["choices"][0]["message"]["content"]
        if trace and trace_span:
            trace.record_llm(trace_span, model, pt, ct)
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=ct, model=model)

    # ---------------- Embedding ----------------
    def embed(self, texts: list[str], trace: TraceRecorder | None = None,
              trace_span: Span | None = None) -> list[list[float]]:
        """
        批量 embedding。批量是为了省 HTTP 往返：一条条发 QPS 很快被打满。
        降级：embed_provider=hash 时用确定性哈希向量（离线可跑、可测试），
              质量差但保证链路完整 —— 这是"可测试性"优先于"效果"的取舍。
        """
        if self.mock or settings.embed_provider == "hash":
            vecs = [_hash_embed(t, settings.embed_dim) for t in texts]
            if trace and trace_span:
                # 粗略估算：中文约 1 token / 1.5 字
                est = sum(max(1, len(t) // 2) for t in texts)
                trace.record_llm(trace_span, "text-embedding-v3", est, 0)
            return vecs

        out: list[list[float]] = []
        bs = settings.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            payload = {"model": settings.embed_model, "input": batch,
                       "dimensions": settings.embed_dim}
            data = self._post("/embeddings", payload)
            # 按 index 排序，防止服务端乱序返回
            items = sorted(data["data"], key=lambda x: x["index"])
            out.extend(item["embedding"] for item in items)
            if trace and trace_span:
                est = sum(max(1, len(t) // 2) for t in batch)
                trace.record_llm(trace_span, settings.embed_model, est, 0)
        return out

    # ---------------- Rerank ----------------
    def rerank(self, query: str, documents: list[str], top_n: int | None = None,
               trace: TraceRecorder | None = None, trace_span: Span | None = None
               ) -> list[tuple[int, float]]:
        """
        重排：返回 [(原下标, 相关性分)]，按分降序。
        注意：百炼的 rerank 是**原生接口**（不是 OpenAI 兼容），路径 /api/v1/services/rerank/text-rerank/text-rerank
        所以这里单独处理，不能复用 _post 的 OpenAI 拼装 —— 这正是"抽象要有边界"的例子。
        """
        top_n = top_n or settings.rerank_top_n
        if self.mock or settings.rerank_provider == "mock":
            # 降级：用关键词重叠度做伪相关分，保证链路可测
            scored = []
            q_terms = set(query.lower())
            for i, d in enumerate(documents):
                overlap = len(q_terms & set(d.lower())) / (len(q_terms) or 1)
                scored.append((i, round(overlap, 4)))
            scored.sort(key=lambda x: -x[1])
            return scored[:top_n]

        # 百炼原生 rerank 接口
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        payload = {
            "model": settings.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = httpx.post(url, json=payload, headers=headers, timeout=settings.llm_timeout)
        if resp.status_code != 200:
            raise classify_http_error(resp.status_code)(f"rerank HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        results = data["output"]["results"]
        if trace and trace_span:
            est = max(1, (len(query) + sum(len(d) for d in documents)) // 2)
            trace.record_llm(trace_span, settings.rerank_model, est, 0)
        return [(r["index"], round(float(r["relevance_score"]), 4)) for r in results]

    # ---------------- mock 实现 ----------------
    def _mock_chat(self, prompt: str, json_mode: bool) -> ChatResult:
        """
        mock 不是随便返回 "hello" —— 它要**足够结构化**，
        让下游引用校验、JSON 解析等逻辑都能被真实测试覆盖。
        """
        pt = max(1, len(prompt) // 2)
        if json_mode:
            # 抽取 prompt 里第一段 [证据N] 的编号，模拟"引用溯源"输出
            text = ('{"answer": "[MOCK] 这是离线 mock 生成的答案（未配置 API Key）。", '
                    '"citations": [], "confidence": 0.5, "refused": false, '
                    '"reason": "mock_mode"}')
        else:
            text = "[MOCK] 离线模式输出（未配置 LLM_API_KEY）。"
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=max(1, len(text) // 2),
                          model="mock", mocked=True)


def _hash_embed(text: str, dim: int) -> list[float]:
    """
    确定性哈希 embedding：把文本映射到固定维度向量。
    用途：无网络 / 无 Key 时保证检索链路可跑、可测（CI 友好）。
    缺点：无语义 —— 所以绝不能把它当"效果"指标，README 必须写清楚。
    """
    vec = [0.0] * dim
    tokens = list(text)
    for i, ch in enumerate(tokens):
        h = int(hashlib.md5(f"{ch}{i}".encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


llm = LLMClient()
