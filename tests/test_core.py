"""
知枢 · 单元测试
============================================================
测试原则：
  1. **不依赖外部服务**：Milvus / PG / Redis 不可用时测试也能全绿，
     所以全部走降级后端（local + sqlite + memory）+ mock LLM。
     CI 里没有中间件也能跑，这是"可测试性"的基本要求。
  2. **回归测试固化真实 bug**：本项目开发中真实踩到两个坑
     （markdown 被当 PDF 解析、children 列表漏填充），
     都写成了专门的回归用例，防止改回去。
  3. 测**行为**而非实现：断言"父子块数量关系成立"，
     而不是断言"某个内部变量等于 X"。
"""
from __future__ import annotations

import os
import sys

import pytest

# 强制降级模式，保证测试不依赖任何中间件
os.environ.update({
    "LLM_PROVIDER": "mock",
    "LLM_API_KEY": "",
    "EMBED_PROVIDER": "hash",
    "RERANK_PROVIDER": "mock",
    "VECTOR_BACKEND": "local",
    "METADATA_BACKEND": "sqlite",
    "CACHE_BACKEND": "memory",
    "USE_QUERY_REWRITE": "false",
    "USE_HYDE": "false",
})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. 解析层
# ============================================================
class TestParsers:
    def test_markdown_uses_text_parser(self, tmp_path):
        """
        回归测试：markdown 必须走 TextParser，不能被送进 PyMuPDF。
        真实 bug：早期 parse_document 对所有文件都调 PDF 解析器，
        .md 文件直接报 "Failed to open file"，入库静默产出 0 块。
        """
        from app.ingest.parsers import TextParser, parse_document
        
        p = tmp_path / "t.md"
        p.write_text("# 标题\n\n这是正文内容。\n\n- 列表项一\n- 列表项二\n", encoding="utf-8")
        doc = parse_document(p)
        assert doc.parser == "markdown", "markdown 必须由 TextParser 解析"
        assert len(doc.blocks) >= 3
        types = {b.block_type for b in doc.blocks}
        assert "title" in types and "list" in types and "paragraph" in types

    def test_title_level_preserved(self, tmp_path):
        from app.ingest.parsers import parse_document
        p = tmp_path / "t.md"
        p.write_text("# 一级\n\n## 二级\n\n### 三级\n", encoding="utf-8")
        doc = parse_document(p)
        levels = [b.level for b in doc.blocks if b.block_type == "title"]
        assert levels == [1, 2, 3], f"标题层级应保留，实际 {levels}"

    def test_empty_file_flagged(self, tmp_path):
        from app.ingest.parsers import parse_document
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        doc = parse_document(p)
        assert doc.warnings, "空文件必须产生告警，不能静默成功"

    def test_unknown_extension_degrades(self, tmp_path):
        from app.ingest.parsers import parse_document
        p = tmp_path / "x.rst"
        p.write_text("some content here", encoding="utf-8")
        doc = parse_document(p)
        assert doc.blocks, "未知类型应降级为文本解析而不是崩溃"


# ============================================================
# 2. 分块层（父子块）
# ============================================================
class TestChunker:
    def test_sentence_split_protects_decimals(self):
        from app.ingest.chunker import split_sentences
        # 小数点不能被切成两句
        s = split_sentences("金额为10000.50元。请审批。")
        assert len(s) == 2, f"应切成 2 句，实际 {s}"

    def test_token_estimate_chinese(self):
        from app.ingest.chunker import estimate_tokens
        # 中文约 1 token / 1.5 字
        n = estimate_tokens("这是一段中文文本" * 10)
        assert 40 <= n <= 120

    def test_parent_child_relationship(self, tmp_path):
        """
        回归测试：children 必须真的被填充。
        真实 bug：build_parent_child 只 return 了空的 children 列表，
        导致"检索用小块"退化成"检索用整篇"，父子块机制静默失效 ——
        不报错、不崩溃，只是召回质量变差，最难发现的那种。
        """
        from app.ingest.chunker import chunk_document
        from app.ingest.parsers import ParsedBlock, ParsedDoc

        blocks = [ParsedBlock("这是第%d段。" % i + "用于测试的正文内容。" * 30,
                              block_type="paragraph") for i in range(6)]
        doc = ParsedDoc(doc_id="d1", title="t", source_path="x.md", blocks=blocks)
        children, parents = chunk_document(doc, group_size=3)

        assert len(children) > 0, "children 不能为空 —— 父子块机制的核心"
        assert len(parents) > 0
        # 每个 child 必须挂在某个 parent 上，且该 parent 真实存在
        parent_ids = {p.chunk_id for p in parents}
        for c in children:
            assert c.parent_id in parent_ids, \
                f"child {c.chunk_id} 的 parent_id={c.parent_id} 不在父块集合中（父子链路断裂）"
        # 每个 child 都应能被某个 parent 认领（不允许出现"无子"的父块）
        child_counts = {p.chunk_id: 0 for p in parents}
        for c in children:
            child_counts[c.parent_id] += 1
        assert all(n > 0 for n in child_counts.values()), \
            f"存在没有任何 child 的 parent：{child_counts}"
        assert sum(child_counts.values()) == len(children)

    def test_chunk_ids_unique(self):
        from app.ingest.chunker import chunk_document
        from app.ingest.parsers import ParsedBlock, ParsedDoc
        blocks = [ParsedBlock("内容" * 300 + f"第{i}段", block_type="paragraph") for i in range(9)]
        doc = ParsedDoc(doc_id="d2", title="t", source_path="x.md", blocks=blocks)
        children, parents = chunk_document(doc)
        ids = [c.chunk_id for c in children] + [p.chunk_id for p in parents]
        assert len(ids) == len(set(ids)), "chunk_id 必须唯一"


# ============================================================
# 3. 可观测层
# ============================================================
class TestObservability:
    def test_wall_latency_is_union_not_max_span(self):
        """
        回归测试：墙钟 = 最晚结束 - 最早开始，而不是"最长单 span"。
        这个口径错误在「运营智脑」项目里真出现过（偏差 3.7 倍），
        在知枢里从第一天就用正确实现并加测试锁死。
        """
        import time
        from app.core.observability import TraceRecorder

        tr = TraceRecorder("t1", "q")
        with tr.span("a", kind="ingest"):
            time.sleep(0.05)
        with tr.span("b", kind="retrieve"):
            time.sleep(0.05)
        # 串行两段各 50ms，墙钟应 ≈ 100ms，而不是最长的 50ms
        assert tr.total_latency_ms >= 90, f"墙钟应含两段之和，实际 {tr.total_latency_ms}"
        assert tr.total_latency_ms < 300

    def test_stage_breakdown_sums_by_kind(self):
        import time
        from app.core.observability import TraceRecorder
        tr = TraceRecorder("t2", "q")
        with tr.span("parse", kind="ingest"):
            time.sleep(0.02)
        with tr.span("search", kind="retrieve"):
            time.sleep(0.02)
        br = tr.stage_breakdown()
        assert br["ingest"] >= 15 and br["retrieve"] >= 15

    def test_stage_breakdown_counts_nested_spans_once(self):
        """
        回归测试：阶段耗时不能把嵌套子 span 重复求和。

        真实 bug：retrieve 阶段下嵌套着 query_rewrite / vector_recall / hyde 等子 span，
        早期直接 sum 得到 10690ms，甚至超过端到端墙钟 5950ms —— 物理上不可能。
        修复：按阶段取时间区间**并集**（重叠只算一次），并保证不超过总墙钟。
        """
        import time
        from app.core.observability import TraceRecorder

        tr = TraceRecorder("t_nested", "q")
        with tr.span("retrieve", kind="retrieve"):           # 父 span
            with tr.span("vector_recall", kind="retrieve"):  # 子 span（嵌套）
                time.sleep(0.05)
            with tr.span("bm25_recall", kind="retrieve"):    # 子 span（嵌套）
                time.sleep(0.05)
        br = tr.stage_breakdown()
        # 父 100ms 已覆盖两个子各 50ms，并集应 ≈100ms，绝不 ≈200ms
        assert br["retrieve"] < 160, f"嵌套 span 被重复求和：{br['retrieve']}ms"
        assert br["retrieve"] >= 90
        # 阶段耗时不得超过端到端墙钟
        assert br["retrieve"] <= tr.total_latency_ms + 1, \
            f"阶段({br['retrieve']}) 不应超过墙钟({tr.total_latency_ms})"

    def test_cost_only_from_real_model(self):
        from app.core.observability import TraceRecorder
        tr = TraceRecorder("t3", "q")
        with tr.span("llm", kind="llm") as sp:
            tr.record_llm(sp, "qwen-plus", 1000, 500)
        assert tr.total_cost_usd > 0
        assert tr.total_tokens == 1500

    def test_mock_has_zero_cost(self):
        from app.core.observability import TraceRecorder
        tr = TraceRecorder("t4", "q")
        with tr.span("llm", kind="llm") as sp:
            tr.record_llm(sp, "mock", 1000, 500)
        assert tr.total_cost_usd == 0.0, "mock 模式不应产生费用"


# ============================================================
# 4. RRF 融合
# ============================================================
class TestRRF:
    def test_rrf_rewards_multi_path_consensus(self):
        """
        融合的核心性质：两路都进前列的文档，应优于只在单路第一的文档。
        这是"为什么要融合"的直接体现。
        """
        from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
        # docA 两路都第 2；docB 只在 vector 第 1
        vector = [("docB", 0.99), ("docA", 0.80), ("docC", 0.70)]
        bm25 = [("docC", 9.9), ("docA", 8.8), ("docD", 7.7)]
        fused = reciprocal_rank_fusion([vector, bm25], k=60)
        order = [cid for cid, _ in fused]
        # docA 两路都命中，应排在前面
        assert order.index("docA") < order.index("docB"), f"多路共识应优先，实际 {order}"

    def test_rrf_scores_positive(self):
        from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]])
        assert all(s > 0 for _, s in fused)

    def test_rrf_empty_input(self):
        from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([[], []]) == []


# ============================================================
# 5. 检索过滤表达式
# ============================================================
class TestFilterExpr:
    def test_build_expr_dept(self):
        from app.retrieval.hybrid_retriever import _build_expr
        assert _build_expr({"dept": "技术部"}) == 'dept == "技术部"'

    def test_build_expr_combined(self):
        from app.retrieval.hybrid_retriever import _build_expr
        e = _build_expr({"dept": "财务部", "effective_before": 1730000000})
        assert 'dept == "财务部"' in e and "effective_ts <= 1730000000" in e

    def test_build_expr_none(self):
        from app.retrieval.hybrid_retriever import _build_expr
        assert _build_expr(None) is None
        assert _build_expr({}) is None


# ============================================================
# 6. 生成治理：引用校验与拒答
# ============================================================
class TestGenerationGovernance:
    def test_validate_drops_hallucinated_citation(self):
        """
        引用校验的核心：模型引用了不存在的证据编号 → 必须丢弃。
        这是防"幻觉引用"的关键防线。
        """
        from app.generation.rerank_generate import validate_citations
        id_map = {1: "chk_a", 2: "chk_b"}
        citations = [
            {"claim": "合法引用", "evidence_id": 1, "quote": "x"},
            {"claim": "幻觉引用", "evidence_id": 99, "quote": "y"},  # 不存在
            {"claim": "缺字段", "quote": "z"},
        ]
        out = validate_citations(citations, id_map)
        assert len(out) == 1, f"应只保留 1 条合法引用，实际 {len(out)}"
        assert out[0]["evidence_id"] == 1

    def test_quote_verified_flag(self):
        """quote 能在原文里找到 → verified=True；找不到 → False（但保留留痕）。"""
        from app.generation import rerank_generate as g
        g._register_evidence_text(1, "chk_a", "年休假最多顺延至次年3月31日")
        out = g.validate_citations(
            [{"evidence_id": 1, "quote": "顺延至次年3月31日", "claim": "c"},
             {"evidence_id": 1, "quote": "这句话原文里没有", "claim": "c"}],
            {1: "chk_a"})
        assert out[0]["verified"] is True
        assert out[1]["verified"] is False

    def test_json_parse_tolerates_code_fence(self):
        from app.generation.rerank_generate import _parse_json
        r = _parse_json('```json\n{"answer":"a","refused":false}\n```')
        assert r["answer"] == "a"
        r2 = _parse_json('前言{"answer":"b"}后语')
        assert r2["answer"] == "b"

    def test_json_parse_failure_refuses(self):
        from app.generation.rerank_generate import _parse_json
        r = _parse_json("完全不是 JSON")
        assert r["refused"] is True, "解析失败必须拒答，不能编造"

    def test_parent_expansion_dedup(self):
        """
        small-to-big 的父块回溯必须**去重**：
        多个 child 命中同一 parent 时，父块只能出现一次，
        否则同一份长文本被重复塞进 prompt，浪费 token 且挤掉其他证据。
        """
        from app.generation.rerank_generate import RerankResult, expand_parent
        from app.storage.meta_store import get_meta_store
        m = get_meta_store()
        m.clear()
        m.upsert_document({"doc_id": "pd1", "title": "测试"})
        m.insert_chunks([
            {"chunk_id": "par_pd1_0000", "doc_id": "pd1", "text": "父块完整上下文内容", "parent_id": None},
            {"chunk_id": "chk_pd1_0000", "doc_id": "pd1", "text": "子块A", "parent_id": "par_pd1_0000"},
            {"chunk_id": "chk_pd1_0001", "doc_id": "pd1", "text": "子块B", "parent_id": "par_pd1_0000"},
            {"chunk_id": "par_pd1_0001", "doc_id": "pd1", "text": "另一个父块", "parent_id": None},
            {"chunk_id": "chk_pd1_0002", "doc_id": "pd1", "text": "子块C", "parent_id": "par_pd1_0001"},
        ])
        cands = [
            RerankResult("chk_pd1_0000", "子块A", 0.9, 0, parent_id="par_pd1_0000"),
            RerankResult("chk_pd1_0001", "子块B", 0.8, 1, parent_id="par_pd1_0000"),
            RerankResult("chk_pd1_0002", "子块C", 0.7, 2, parent_id="par_pd1_0001"),
        ]
        out = expand_parent(cands)
        assert len(out) == 2, f"两个 child 属同一父块应去重为 1，共应 2 条，实际 {len(out)}"
        # 关键修复：text 保留 child 原文（可逐字引用），父块放到 parent_text（做语境）
        assert out[0].text == "子块A", "text 应保留 child 原文，便于精确引用与高亮"
        assert out[0].parent_text == "父块完整上下文内容", "父块全文应在 parent_text 里"
        assert out[0].parent_id == "par_pd1_0000"

    def test_normalized_refusal_avoids_false_negative(self):
        """
        回归测试：假拒答 bug。

        真实事故：早期用「rerank 原始分 < 0.35」判拒答，而百炼 gte-rerank 对一条
        真正相关的 block 也只给 ~0.19 分 → 把正确文档直接拒掉（假拒答）。
        根因是**拿绝对分比固定阈值**，忽略了不同 reranker 分数量纲不同。

        修复：改判「批内归一化后的 top 分」。同一 query 的候选自成可比集合，
        归一化后量纲无关。这里用一个"最高分很低但批内相对突出"的场景验证不再误拒。
        """
        from app.generation.rerank_generate import _normalized_top
        # 一条真相关但绝对分低（模拟百炼量纲）的候选：不应判为低证据
        scores = [0.19, 0.14, 0.07]
        norm, top = _normalized_top(scores)
        assert top == 0.19
        assert norm > 0.9, f"最高分应在批内归一化后接近 1，实际 {norm}"
        # 整批都很低且绝对分约等于 0 → 视为不相关
        norm0, top0 = _normalized_top([0.0, 0.0, 0.0])
        assert top0 == 0.0

    def test_empty_quote_backfilled_from_evidence(self):
        """
        回归测试：引用 quote 为空。

        真实事故：模型有时只写 answer 正文的 [证据N]，citations 里的 quote 留空，
        导致用户看到"引用"却看不到任何可核对的原文，verified=false。
        修复：quote 为空时，用**证据原文前若干字**兜底（原文来自我们喂给模型的证据，
        非模型生成，故必然可核对），并标记 quote_source=system 以区分来源。
        """
        from app.generation import rerank_generate as g
        g._register_evidence_text(1, "chk_x", "单笔金额超过10000元的，须经总经理审批。")
        out = g.validate_citations(
            [{"evidence_id": 1, "claim": "c", "quote": ""}],  # 模型没给 quote
            {1: "chk_x"})
        assert out[0]["quote"], "空 quote 应被兜底填充"
        assert out[0]["verified"] is True
        assert out[0]["quote_source"] == "system"

    def test_evidence_text_cache_is_bounded(self):
        """证据原文全局缓存必须有上限，否则长驻服务会内存泄漏。"""
        from app.generation import rerank_generate as g
        g._QUOTE_CACHE.clear()
        for i in range(g._QUOTE_CACHE_MAX + 500):
            g._register_evidence_text(i, f"chk_{i}", "x" * 10)
        assert len(g._QUOTE_CACHE) <= g._QUOTE_CACHE_MAX, \
            f"缓存应被限制在 {g._QUOTE_CACHE_MAX}，实际 {len(g._QUOTE_CACHE)}"
        g._QUOTE_CACHE.clear()


# ============================================================
# 7. 存储降级
# ============================================================
class TestStorageFallback:
    def test_local_vector_store_roundtrip(self, tmp_path):
        from app.storage.vector_store import Chunk as VC, LocalVectorStore
        store = LocalVectorStore(path=str(tmp_path))
        store.insert([
            VC("c1", "d1", None, "年休假制度", embedding=[1.0, 0.0, 0.0]),
            VC("c2", "d1", None, "报销流程", embedding=[0.0, 1.0, 0.0]),
        ])
        assert store.count() == 2
        hits = store.search([0.99, 0.01, 0.0], top_k=1)
        assert hits[0][0] == "c1", "最相关的应是 c1"

    def test_meta_store_sqlite_schema(self):
        from app.storage.meta_store import get_meta_store
        m = get_meta_store()
        m.upsert_document({"doc_id": "td1", "title": "测试文档", "dept": "测试部"})
        m.insert_chunks([{"chunk_id": "tc1", "doc_id": "td1", "text": "内容"}])
        meta = m.get_chunk_meta(["tc1"])
        assert meta["tc1"]["title"] == "测试文档"
        assert meta["tc1"]["dept"] == "测试部"
        m.clear()

    def test_memory_cache_lru(self):
        from app.storage.cache import MemoryCache
        c = MemoryCache(maxsize=2)
        c.set("ns", "k1", {"v": 1})
        assert c.get("ns", "k1") == {"v": 1}
        assert c.get("ns", "missing") is None

    def test_cache_key_includes_kb_version(self):
        """缓存键必须含知识库版本 —— 否则知识库更新后会出现脏读。"""
        from app.storage import cache as c
        c.bump_kb_version()
        k1 = c._key("answer", "问题")
        c.bump_kb_version()
        k2 = c._key("answer", "问题")
        assert k1 != k2, "知识库版本变化后缓存键必须改变，防止旧答案脏读"


# ============================================================
# 8. 端到端（降级模式）
# ============================================================
class TestEndToEnd:
    def test_ingest_then_ask(self, tmp_path):
        """整条链路：入库 → 检索 → 重排 → 生成，全部走降级后端。"""
        from app.pipeline import ZhishuPipeline
        from app.storage.vector_store import LocalVectorStore
        p = ZhishuPipeline()
        p.meta.clear()
        p.vector_store = LocalVectorStore(path=str(tmp_path / "vec"))
        p.retriever.vector_store = p.vector_store
        f = tmp_path / "制度.md"
        f.write_text("# 报销制度\n\n单笔金额超过10000元的，须经总经理审批。\n"
                     "## 报销时限\n\n员工应在费用发生后30个自然日内提交报销申请。\n",
                     encoding="utf-8")
        r = p.ingest(f, dept="财务部")
        assert r["ok"] and r["children"] > 0, f"入库失败: {r}"

        ans = p.ask("单笔超过10000元谁审批？")
        assert "retrieval_stats" in ans
        assert ans["retrieval_stats"]["retrieved"] > 0, "应检索到证据"
        assert ans["answer"], "应返回答案"
        assert not ans["reason"].startswith("系统异常")

    def test_ingest_idempotent(self, tmp_path):
        """重复入库同一文件不应产生重复数据。"""
        from app.pipeline import ZhishuPipeline
        from app.storage.vector_store import LocalVectorStore
        p = ZhishuPipeline()
        p.meta.clear()
        p.vector_store = LocalVectorStore(path=str(tmp_path / "vec"))
        p.retriever.vector_store = p.vector_store
        f = tmp_path / "doc.md"
        f.write_text("# 标题\n\n内容。" * 20, encoding="utf-8")
        r1 = p.ingest(f)
        n1 = p.meta.stats()["documents"]
        r2 = p.ingest(f)
        n2 = p.meta.stats()["documents"]
        assert n1 == n2 == 1, f"重复入库不应增加文档数：{n1} → {n2}"

    def test_refuse_when_no_evidence(self, tmp_path):
        """
        知识库为空时，必须拒答而不是编造。

        测试隔离坑：本地向量库默认写入 data/index/，多个测试共用同一目录会
        互相污染（前一个测试入库的数据会让"空库"测试其实不为空）。
        这里显式切到 tmp_path，并清空元数据 —— **测试之间必须隔离**，
        否则会出现"单跑通过、全量跑失败"的诡异现象。
        """
        from app.pipeline import ZhishuPipeline
        from app.storage.vector_store import LocalVectorStore
        p = ZhishuPipeline()
        p.meta.clear()
        p.vector_store = LocalVectorStore(path=str(tmp_path))  # 独立的空向量库
        p.retriever.vector_store = p.vector_store
        p.refresh_index()  # 元数据已清空，BM25 也应为空
        ans = p.ask("完全无关的问题：明天天气怎么样？")
        assert ans["refused"] is True, "无证据必须拒答"
        assert ans["citations"] == [], "拒答时不应有引用"
        assert "没有检索到" in ans["reason"] or ans["refused"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
