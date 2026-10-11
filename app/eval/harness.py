"""
知枢 · 离线评测（评测集回归 + LLM-as-Judge + CI 门禁）
============================================================
简历原文：「Ragas + LLM-as-Judge 离线评测接入 CI 发布门禁（指标不达标即阻断）」

评测什么（RAG 三维度）：
  1. **忠实度 Faithfulness**：答案是否**完全**由检索到的证据支撑（防幻觉）。
  2. **答案相关性 Answer Relevancy**：答案是否**切题**（防答非所问）。
  3. **上下文召回率 Context Recall**：正确回答所需的信息，是否**被检索到**。

为什么评测必须对接 CI（面试核心）：
  → 没有门禁的评测就是"看一眼报告就过去了"。RAG 系统改一个 prompt、
    换个模型、调个 chunk_size，效果都可能退化；如果每次都靠人肉判断，
    迟早会带着退化上线。把阈值写进 CI，**退化就自动阻断合并** ——
    这才是"工程化"和"做了个评测脚本"的分界线。

诚实标注（重要）：
  · 本模块默认使用一个**小型内置评测集**（基于 data/corpus 的样例文档），
    用于跑通流程与 CI 门禁。它不是权威基准，数字只反映"本项目在自建集上的表现"。
  · 不引用任何外部论文/榜单数字做对比，避免无法自证的宣称。
  · 无 LLM Key 时，judge 自动降级为**启发式打分**（关键词重叠 / 引用覆盖率），
    并在报告中明确标注"启发式（非 LLM 判定）"。
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.llm_client import llm

logger = logging.getLogger("zhishu.eval")


# ============================================================
# 评测集：问题 + 参考答案要点 + 期望证据关键词
# 设计原则：问题要覆盖"语义型"与"编号型"两类，才能检验混合检索的价值。
# ============================================================
EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_set.jsonl"

DEFAULT_EVAL_SET = [
    {"q": "年休假最多可以顺延到什么时候？", "expect": "次年3月31日", "keywords": ["3月31日", "顺延"]},
    {"q": "Work 满 10 年不满 20 年的员工年休假多少天？", "expect": "10天", "keywords": ["10天", "10年"]},
    {"q": "单笔报销金额超过 10000 元需要谁审批？", "expect": "总经理", "keywords": ["总经理", "10000"]},
    {"q": "报销申请应该在费用发生后多少天内提交？", "expect": "30个自然日内", "keywords": ["30", "自然日"]},
    {"q": "同一笔费用最多可以退回补充几次？", "expect": "2次", "keywords": ["2次", "退回"]},
    {"q": "员工每月迟到累计超过几次会影响绩效？", "expect": "3次", "keywords": ["3次", "迟到"]},
    {"q": "女职工生育可以享受多少天产假？", "expect": "158天", "keywords": ["158"]},
    {"q": "出差住宿在北上广深的每晚限额是多少？", "expect": "500元", "keywords": ["500"]},
    # 负样本：知识库里没有答案，系统应正确拒答
    {"q": "公司年会一般在哪个酒店举办？", "expect": "REFUSE", "keywords": [], "should_refuse": True},
    {"q": "公司的股票代码是多少？", "expect": "REFUSE", "keywords": [], "should_refuse": True},
]


@dataclass
class EvalSample:
    q: str
    expect: str
    keywords: list[str] = field(default_factory=list)
    should_refuse: bool = False
    answer: str = ""
    refused: bool = False
    citations: list = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)


# ============================================================
# 单指标实现
# ============================================================
def answer_relevancy(sample: EvalSample) -> float:
    """
    答案相关性：期望关键词有多少命中了答案。
    若该题应拒答，则"成功拒答"给满分。
    """
    if sample.should_refuse:
        return 1.0 if sample.refused else 0.0
    if not sample.keywords:
        return 1.0 if sample.answer else 0.0
    hit = sum(1 for k in sample.keywords if k in sample.answer)
    return round(hit / len(sample.keywords), 4)


def faithfulness(sample: EvalSample) -> float:
    """
    忠实度：答案里的引用，有多少是"经过校验"的（verified=True）。
    没有引用的非拒答答案，忠实度按 0 计 —— 因为无法证明它来自证据。
    """
    if sample.should_refuse:
        return 1.0 if sample.refused else 0.0
    if not sample.citations:
        return 0.0
    ok = sum(1 for c in sample.citations if c.get("verified"))
    return round(ok / len(sample.citations), 4)


def context_recall(sample: EvalSample, retrieved_texts: list[str]) -> float:
    """
    上下文召回率：期望关键词有多少出现在**检索到的证据原文**里。
    注意这是"检索层"的指标，与答案层无关 —— 它能区分"没检索到"和"检索到但答错"。
    """
    if sample.should_refuse or not sample.keywords:
        return 1.0
    blob = "\n".join(retrieved_texts)
    hit = sum(1 for k in sample.keywords if k in blob)
    return round(hit / len(sample.keywords), 4)


def llm_judge(sample: EvalSample) -> float:
    """
    LLM-as-Judge：让模型给"答案是否准确回答了问题"打分（0/0.5/1）。
    降级：无 Key 时退回启发式（用 answer_relevancy 代替），并标记 judged_by。
    """
    if llm.mock or sample.should_refuse:
        return answer_relevancy(sample)
    prompt = (
        "你是严格的评审。请判断【答案】是否准确回答了【问题】，只依据常识与答案本身。\n"
        "打分规则：完全正确=1；部分正确或缺关键信息=0.5；错误或答非所问=0。\n"
        "只输出一个 JSON：{\"score\": 0|0.5|1, \"reason\": \"...\"}\n\n"
        f"【问题】{sample.q}\n【答案】{sample.answer}\n"
    )
    try:
        res = llm.chat(prompt, temperature=0.0, max_tokens=256, json_mode=True)
        from app.generation.rerank_generate import _parse_json
        parsed = _parse_json(res.text)
        score = parsed.get("score")
        if score is None:
            # judge 输出不合规：明确降级并记录，避免"静默给 0 分"污染指标
            logger.warning("judge 未返回 score 字段：%r，降级启发式", res.text[:120])
            return answer_relevancy(sample)
        return float(score)
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge 调用失败，降级启发式：%s", exc)
        return answer_relevancy(sample)


# ============================================================
# 主流程
# ============================================================
def load_eval_set() -> list[dict]:
    if EVAL_SET_PATH.exists():
        rows = [json.loads(l) for l in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            return rows
    return DEFAULT_EVAL_SET


def run_eval(limit: int | None = None, use_judge: bool = True,
             write_report: bool = True) -> dict[str, Any]:
    """跑一轮评测，返回指标汇总；可选写出 Markdown 报告。"""
    from app.pipeline import get_pipeline

    pipe = get_pipeline()
    cases = load_eval_set()
    if limit:
        cases = cases[:limit]

    samples: list[EvalSample] = []
    for case in cases:
        s = EvalSample(q=case["q"], expect=case.get("expect", ""),
                       keywords=case.get("keywords", []),
                       should_refuse=case.get("should_refuse", False))
        try:
            result = pipe.ask(s.q, use_cache=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("评测用例执行失败：%s", exc)
            result = {"answer": "", "refused": True, "citations": [],
                      "sources": [], "latency_ms": 0, "cost_usd": 0}
        s.answer = result.get("answer", "")
        s.refused = bool(result.get("refused"))
        s.citations = result.get("citations", [])
        s.latency_ms = result.get("latency_ms", 0.0)
        s.cost_usd = result.get("cost_usd", 0.0)
        retrieved = [src.get("text", "") for src in result.get("sources", [])]
        s.scores = {
            "answer_relevancy": answer_relevancy(s),
            "faithfulness": faithfulness(s),
            "context_recall": context_recall(s, retrieved),
            "llm_judge": llm_judge(s) if use_judge else answer_relevancy(s),
        }
        samples.append(s)

    def avg(key: str) -> float:
        vals = [s.scores[key] for s in samples]
        return round(statistics.mean(vals), 4) if vals else 0.0

    summary = {
        "n": len(samples),
        "judged_by": "heuristic(mock)" if llm.mock else "llm",
        "metrics": {
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "context_recall": avg("context_recall"),
            "llm_judge": avg("llm_judge"),
        },
        "latency_ms": {
            "mean": round(statistics.mean([s.latency_ms for s in samples]), 1) if samples else 0,
            "p95": _p95([s.latency_ms for s in samples]),
        },
        "total_cost_usd": round(sum(s.cost_usd for s in samples), 6),
        "refuse": {
            "expected": sum(1 for s in samples if s.should_refuse),
            "correct": sum(1 for s in samples if s.should_refuse and s.refused),
        },
        "details": [
            {"q": s.q, "answer": s.answer[:160], "refused": s.refused,
             "scores": s.scores, "citations": len(s.citations)}
            for s in samples
        ],
    }
    if write_report:
        write_markdown_report(summary, samples)
    return summary


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(-(-0.95 * len(s) // 1)) - 1)], 1)


def write_markdown_report(summary: dict, samples: list[EvalSample]) -> Path:
    out = Path(settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_report.md"
    m = summary["metrics"]
    lines = [
        "# 知枢 · 评测报告",
        "",
        f"> 评测集：内置自建集（{summary['n']} 题，涵盖语义型 / 编号型 / 负样本）",
        f"> 判定方式：{summary['judged_by']}",
        "> **说明**：以下数字仅反映本项目在自建评测集上的表现，非权威基准，不代表通用能力。",
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---|---|",
        f"| 忠实度 Faithfulness | {m['faithfulness']:.3f} | 答案是否有可验证的证据支撑（防幻觉） |",
        f"| 答案相关性 Answer Relevancy | {m['answer_relevancy']:.3f} | 答案是否切题 |",
        f"| 上下文召回 Context Recall | {m['context_recall']:.3f} | 所需信息是否被检索到（检索层指标） |",
        f"| LLM-as-Judge | {m['llm_judge']:.3f} | 综合正确性评分 |",
        "",
        f"- 平均延迟：{summary['latency_ms']['mean']} ms（P95 {summary['latency_ms']['p95']} ms）",
        f"- 本轮总成本：${summary['total_cost_usd']}",
        f"- 拒答正确率：{summary['refuse']['correct']}/{summary['refuse']['expected']}"
        "（负样本必须拒答，答了就是幻觉）",
        "",
        "## 二、逐题明细",
        "",
        "| 问题 | 是否拒答 | 忠实度 | 相关性 | 召回 | Judge | 引用数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in summary["details"]:
        sc = d["scores"]
        lines.append(
            f"| {d['q'][:28]} | {'是' if d['refused'] else '否'} | "
            f"{sc['faithfulness']:.2f} | {sc['answer_relevancy']:.2f} | "
            f"{sc['context_recall']:.2f} | {sc['llm_judge']:.2f} | {d['citations']} |")
    lines += ["", "## 三、复现方式", "",
              "```bash", "python -m scripts.run_eval --limit 10", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("评测报告已写入 %s", path)
    return path


def check_gate(summary: dict, thresholds: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """
    CI 门禁：所有指标都必须 ≥ 阈值，否则判定失败（阻断合并）。
    面试考点：为什么门禁要"全部通过"而不是"平均分达标"？
      → 平均分会掩盖短板。忠实度 0.5 + 相关性 1.0 平均 0.75 看着还行，
        但 0.5 的忠实度意味着**每两个答案就有一个在编**，这是不可接受的。
        所以按**单项下限**卡，而不是看总分。
    """
    th = thresholds or {"faithfulness": 0.6, "answer_relevancy": 0.6,
                        "context_recall": 0.6, "llm_judge": 0.6}
    failures = []
    for k, v in th.items():
        got = summary["metrics"].get(k, 0.0)
        if got < v:
            failures.append(f"{k}: {got:.3f} < 阈值 {v:.2f}")
    # 拒答正确率必须 100%（负样本不能有任何幻觉）
    rd = summary.get("refuse", {})
    if rd.get("expected") and rd.get("correct", 0) < rd["expected"]:
        failures.append(f"拒答正确率 {rd['correct']}/{rd['expected']}，负样本出现幻觉")
    return len(failures) == 0, failures
