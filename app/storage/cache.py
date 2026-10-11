"""
知枢 · 缓存层（Redis，带进程内 LRU 降级）
============================================================
面试考点：RAG 为什么要缓存？缓存什么？

  1. 缓存 **embedding**：同一段文本反复 embed 是纯浪费（向量是确定性的）。
  2. 缓存 **检索结果**：热点问题（如"报销标准是多少"）会被反复问。
  3. 缓存 **最终答案**：完全相同的 question 直接返回，0 token 成本。

  但不能无脑缓存最终答案 —— 因为知识库更新了，旧答案就错了。
  所以：
    · key 里带上"知识库版本号"，索引一变，缓存全失效（version-based invalidation）
    · TTL 兜底（settings.cache_ttl_seconds）

  这个"缓存键要包含数据版本"是很多线上事故的根因，面试高频。

降级：Redis 不可用时用进程内 LRU —— 单机开发不装 Redis 也能跑。
      注意多副本下内存缓存不共享，所以生产必须用 Redis。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.core.config import settings
from app.core.observability import metrics

logger = logging.getLogger("zhishu.cache")

# 知识库版本号：每次重建索引就 +1，从而让所有旧缓存失效
#
# 面试考点（踩过的真实坑）：
#   早期用**进程内全局计数器**做版本号，单机单进程没问题，但一旦出现两种情形就翻车：
#     ① 多进程部署（uvicorn --workers N / 多副本）：每个进程各持一份计数器，
#        A 进程 bump 后 B 进程的缓存键仍指向旧版本 → 脏读；
#     ② 进程重启：计数器归零，又把版本号变回 v0/v1，于是**历史遗留的旧缓存
#        重新可见**（正是本项目的现场现象）。
#   正确做法：把版本号当成**共享状态**存进 Redis（KV 的单一事实来源），
#   进程内只做后备（无 Redis 时才用内存计数器）。
_KV_VERSION = 0
_KV_KEY = "zhishu:kb_version"


def _kv_redis():  # 惰性拿一个轻量 redis 连接，仅用于版本号读写
    try:
        import redis
        c = redis.from_url(settings.redis_url, decode_responses=True,
                           socket_connect_timeout=2)
        c.ping()
        return c
    except Exception:  # noqa: BLE001
        return None


def bump_kb_version() -> str:
    """
    重建索引后调用：让全部旧缓存立即失效，避免"知识库更新了答案还旧"的脏读。

    坑一：早期实现用 `int(time.time())` 做版本号 —— 同一秒内连续调用两次
    （例如批量入库多个文档）会得到**相同**的版本号，缓存不会被失效。
    坑二：用进程内全局计数器 —— 多进程/重启后不共享，会重新看到旧缓存。
    正确做法：**Redis 单调递增计数器**（共享、持久），无 Redis 时退回内存计数器。
    """
    global _KV_VERSION
    r = _kv_redis()
    if r is not None:
        try:
            v = int(r.incr(_KV_KEY))
            _KV_VERSION = v
            logger.info("知识库版本号更新为 v%d（Redis，旧缓存全部失效）", v)
            return kb_version()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 版本号自增失败，退回内存计数器：%s", exc)
    _KV_VERSION += 1
    logger.info("知识库版本号更新为 v%d（内存，旧缓存全部失效）", _KV_VERSION)
    return kb_version()


def kb_version() -> str:
    global _KV_VERSION
    if _KV_VERSION == 0:
        r = _kv_redis()
        if r is not None:
            try:
                v = r.get(_KV_KEY)
                _KV_VERSION = int(v) if v else 0
            except Exception:  # noqa: BLE001
                pass
    return f"v{_KV_VERSION}"


def _key(namespace: str, raw: str) -> str:
    """缓存键 = 命名空间 + 知识库版本 + 内容哈希。版本入键是防脏读的关键。"""
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"zhishu:{_KV_VERSION}:{namespace}:{h}"


class Cache:
    """统一缓存接口。get/set 语义与 dict 一致，值为任意 JSON 可序列化对象。"""

    def get(self, namespace: str, raw: str) -> Any | None:
        raise NotImplementedError

    def set(self, namespace: str, raw: str, value: Any, ttl: int | None = None) -> None:
        raise NotImplementedError


class RedisCache(Cache):
    def __init__(self) -> None:
        import redis
        self.client = redis.from_url(settings.redis_url, decode_responses=True,
                                     socket_connect_timeout=3)
        self.client.ping()

    def get(self, namespace: str, raw: str) -> Any | None:
        try:
            v = self.client.get(_key(namespace, raw))
            if v is None:
                metrics.cache_misses += 1
                return None
            metrics.cache_hits += 1
            return json.loads(v)
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存读取失败（降级为未命中）: %s", exc)
            return None

    def set(self, namespace: str, raw: str, value: Any, ttl: int | None = None) -> None:
        try:
            self.client.set(_key(namespace, raw),
                            json.dumps(value, ensure_ascii=False),
                            ex=ttl or settings.cache_ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("缓存写入失败（忽略）: %s", exc)


class MemoryCache(Cache):
    """
    进程内 LRU。面试考点：它和 Redis 的区别？
      · 零运维、纳秒级、但多副本不共享、重启即丢。
      · 适合单机开发 / 单副本部署；一旦水平扩容就必须换 Redis，
        否则会出现"同一个问题在 A 副本命中、B 副本未命中"的不一致。
    """

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, namespace: str, raw: str) -> Any | None:
        k = _key(namespace, raw)
        with self._lock:
            item = self._store.get(k)
            if item is None:
                metrics.cache_misses += 1
                return None
            expire_at, value = item
            if expire_at < time.time():
                self._store.pop(k, None)
                metrics.cache_misses += 1
                return None
            self._store.move_to_end(k)
            metrics.cache_hits += 1
            return value

    def set(self, namespace: str, raw: str, value: Any, ttl: int | None = None) -> None:
        k = _key(namespace, raw)
        with self._lock:
            self._store[k] = (time.time() + (ttl or settings.cache_ttl_seconds), value)
            self._store.move_to_end(k)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is not None:
        return _cache
    if not settings.enable_cache:
        logger.warning("缓存已关闭（enable_cache=false）")
        _cache = MemoryCache()  # 空实现也走同一接口，业务层无需分支
        return _cache
    if settings.cache_backend == "memory":
        _cache = MemoryCache()
        logger.info("缓存后端：进程内 LRU")
        return _cache
    try:
        _cache = RedisCache()
        logger.info("缓存后端：Redis @ %s", settings.redis_url)
        return _cache
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用（%s），降级到进程内 LRU", exc)
        _cache = MemoryCache()
        return _cache
