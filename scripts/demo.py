#!/usr/bin/env python3
"""
知枢 · 端到端演示（无需起服务，一条命令跑通"入库 → 检索 → 生成 → 引用校验"全链路）

用法：
  python -m scripts.demo                # 入库样例语料 + 跑 5 个问题（含 1 个"拒答"）
  python -m scripts.demo --no-ingest    # 复用已有索引，只跑问答
  python -m scripts.demo --rebuild      # 清空后重建索引再跑

说明：
  - 未配置任何 Key 时自动进入 mock 模式，链路照样跑通（答案会标注为 mock）。
  - 配置了百炼 Key 则走真实模型，能看到真实引用与成本统计。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import get_pipeline  # noqa: E402

CORPUS = Path("data/corpus")

DEMO_QUESTIONS = [
    ("在知识库内", "年休假可以顺延到什么时候？"),
    ("在知识库内", "采购一台 5 万元的设备需要谁审批？"),
    ("在知识库内", "迟到超过几次会影响绩效考核？"),
    ("多跳推理",   "我工作满 12 年，今年年假有几天，能不能留到明年？"),
    ("知识库外",   "公司年会的抽奖奖品预算大概是多少？"),  # 期望：拒答
]


def hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description="知枢 端到端演示")
    ap.add_argument("--no-ingest", action="store_true", help="跳过入库")
    ap.add_argument("--rebuild", action="store_true", help="入库前清空")
    args = ap.parse_args()

    pipe = get_pipeline()

    hr("① 入库")
    if not args.no_ingest:
        if args.rebuild:
            pipe.vector_store.drop(); pipe.meta.clear(); pipe.refresh_index()
            print("已清空旧索引")
        files = sorted(CORPUS.glob("*.md")) + sorted(CORPUS.glob("*.txt"))
        for f in files:
            r = pipe.ingest(f)
            flag = "✓" if r.get("ok") else "✗"
            print(f"  {flag} {f.name:<34} 子块={r.get('children', '-')} "
                  f"父块={r.get('parents', '-')} trace={r.get('trace_id')}")
    else:
        print("  （跳过，复用已有索引）")

    print("\n  知识库状态：", json.dumps(
        {k: pipe.stats()[k] for k in ("vector_backend", "vector_count",
                                      "metadata_backend", "cache_backend", "llm_mock")},
        ensure_ascii=False))

    hr("② 问答全链路")
    if pipe.stats().get("llm_mock"):
        print("  ⚠ 当前为 mock 模式（未配置模型 Key）：答案与拒绝行为均为占位，")
        print("    仅用于验证链路连通性，**不代表真实效果**。配置 Key 后重跑即为真实结果。\n")
    for tag, q in DEMO_QUESTIONS:
        r = pipe.ask(q)
        print(f"\n【{tag}】Q: {q}")
        if r.get("refused"):
            print(f"  → [拒答] {r.get('reason') or '证据不足'}")
            print(f"  （这是正确行为：知识库里确实没有这条信息）")
            continue
        print(f"  → A: {r['answer']}")
        if r.get("citations"):
            print(f"  引用 {len(r['citations'])} 条：")
            for c in r["citations"]:
                badge = "✓已核对原文" if c.get("verified") else "✗未核对"
                num = c.get("evidence_id")
                healed = "  (编号已自愈)" if c.get("corrected") else ""
                snip = (c.get("quote") or c.get("text") or "")[:48].replace("\n", " ")
                print(f"    [证据{num}] {badge}{healed}  {snip}...")
        st = r.get("retrieval_stats", {})
        print(f"  检索：候选 {st.get('retrieved')} → 重排 {st.get('reranked')}"
              f"  |  置信度 {r.get('confidence')}")
        print(f"  耗时 {r.get('latency_ms')}ms  阶段 "
              f"{json.dumps(r.get('stage_breakdown_ms', {}), ensure_ascii=False)}"
              f"  成本 ${r.get('cost_usd', 0):.6f}")
        print(f"  trace_id={r.get('trace_id')}  （可到 data/traces/ 查看完整 Span 树）")

    hr("③ 缓存验证（先强制不走缓存，再同一问题提问应命中缓存）")
    q = "年休假可以顺延到什么时候？"
    # 第一次：绕过缓存，拿到真实耗时
    r1 = pipe.ask(q, use_cache=False)
    # 第二次：允许走缓存
    r2 = pipe.ask(q, use_cache=True)
    print(f"  绕过缓存（use_cache=False）cached={r1.get('cached')} 耗时 {r1.get('latency_ms')}ms")
    print(f"  再次提问（use_cache=True）cached={r2.get('cached')} 耗时 {r2.get('latency_ms')}ms")
    if r2.get("cached"):
        print(f"  结论：第二次命中缓存，直接返回，跳过检索与生成，"
              f"省下约 {round(r1.get('latency_ms', 0) - r2.get('latency_ms', 0), 1)}ms。")
    else:
        print("  注意：第二次未命中缓存（可能因答案为空或策略未缓存）。")

    print("\n完成。下一步可跑：python -m scripts.run_eval --gate  验证质量门禁。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
