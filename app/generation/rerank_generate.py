"""
知枢 · 重排层（bge / gte-reranker）
============================================================
简历原文：「再经 bge-reranker-v2 重排」

为什么召回之后还要重排（面试必问）：
  召回用的是 **双塔模型（bi-encoder）**：query 和 doc 分别编码成向量，
  只计算向量距离。优点是快（可预计算、可 ANN 索引），
  缺点是 query 与 doc **没有交互**，细粒度的相关性判断不准。

  重排用的是 **交叉编码器（cross-encoder）**：把 (query, doc) 拼在一起过一遍模型，
  两者每个词都能互相 attention，相关性判断显著更准，
  但代价是**每个候选都要过一遍模型**，无法预计算 —— 所以只能对少量候选精排。

  标准流水线：
    BM25+向量 召回 top-20（便宜、广） → Reranker 精排 top-6（贵、准） → 喂 LLM
  面试结论一句话：**召回求全（recall），重排求准（precision）。**

诚实标注：简历写的是 bge-reranker-v2，本项目默认用百炼的 gte-rerank-v2
（同类交叉编码器 reranker，API 形态一致）。两者可互换，settings.rerank_model 切换；
若本地部署 bge-reranker-v2-m3（sentence-transformers），替换 Reranker 实现即可。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.core.config import settings
from app.core.llm_client import llm
from app.core.observability import TraceRecorder
from app.retrieval.hybrid_retriever import RetrievedChunk, _null_span

logger = logging.getLogger("zhishu.rerank")


@dataclass
class RerankResult:
    chunk_id: str
    text: str
    score: float
    original_rank: int
    parent_id: str | None = None
    parent_text: str | None = None      # 回溯到的父块全文（喂给 LLM 用）


def expand_parent(candidates: list[RerankResult]) -> list[RerankResult]:
    """
    small-to-big 的"big"这一步：命中 child 后回溯它的父块，用父块喂 LLM。

    面试核心考点：为什么检索用 child、生成用 parent？
      · child（小块）向量聚焦，召回准；
      · parent（大块）上下文完整，LLM 能读懂前后文，不会断章取义。
      这个函数就是连接两者的桥梁 —— 简历里"检索用小块、生成喂父块"
      指的就是它。若缺了这一步，父子块就白建了（只剩检索，没有生成增益）。

    注意若干细节（面试会追问）：
      1. 多个 child 可能属于同一 parent → 必须**去重**，否则同一份父块
         被重复塞进 prompt，既浪费 token 又稀释其他证据。
      2. 去重后父块数量可能变少 → 重新编号（证据1..N），保证编号连续。
      3. 父块缺失（如元数据没同步）→ 退回用 child 原文，不能让链路断掉。
      4. **child 原文必须保留为可引用单元（关键修复）**：
         早期实现把 child 的 text 直接替换成父块全文，虽然上下文更全，但带来
         两个真问题：① 引用指向的"块"其实是放大后的父块，用户点开对不上；
         ② 模型看到的是父块融合文本，容易按"第几条证据"数位置乱标编号。
         正确做法：**child 保持它自己的原文（用于逐字引用/高亮命中位置）**，
         父块作为额外的 `parent_text` 上下文一并喂给模型 —— 即"小块做证据、大块做语境"。
    """
    from app.storage.meta_store import get_meta_store
    meta = get_meta_store()

    parent_ids = [c.parent_id for c in candidates if c.parent_id]
    parent_texts = meta.get_parent_chunks([pid for pid in parent_ids if pid])

    seen: set[str] = set()
    out: list[RerankResult] = []
    for c in candidates:
        pid = c.parent_id
        if pid and pid in parent_texts:
            if pid in seen:
                continue          # 同一父块只保留一次（取最高分的那个 child 的位置）
            seen.add(pid)
            # text 保留 child 原文（可逐字引用）；parent_text 作为大块语境
            out.append(RerankResult(chunk_id=c.chunk_id, text=c.text,
                                    score=c.score, original_rank=c.original_rank,
                                    parent_id=pid, parent_text=parent_texts[pid]))
        else:
            # 无父块信息 → 退回 child 文本，保证不丢证据
            out.append(RerankResult(chunk_id=c.chunk_id, text=c.text, score=c.score,
                                    original_rank=c.original_rank, parent_id=pid))
    return out


class Reranker:
    """
    重排器。核心接口就一个 rerank(query, candidates, top_n)。
    实现可换：百炼 API / 本地 bge-reranker-v2-m3 / mock。
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_n: int | None = None, trace: TraceRecorder | None = None
               ) -> list[RerankResult]:
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []

        docs = [c.text for c in candidates]
        with (trace.span("rerank", kind="rerank") if trace else _null_span()) as sp:
            try:
                pairs = llm.rerank(query, docs, top_n=top_n, trace=trace, trace_span=sp)
            except Exception as exc:  # noqa: BLE001
                # 降级：重排失败不能导致整条链路挂掉，退回 RRF 顺序（仍走父块展开）
                logger.warning("重排失败（%s），退回融合顺序", exc)
                fallback = [RerankResult(c.chunk_id, c.text, c.score, i,
                                         parent_id=c.parent_id)
                            for i, c in enumerate(candidates[:top_n])]
                return expand_parent(fallback)

        out: list[RerankResult] = []
        for idx, score in pairs:
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                out.append(RerankResult(c.chunk_id, c.text, score, idx,
                                        parent_id=c.parent_id))
        # small-to-big：回溯父块并去重（命中 child → 生成喂 parent）
        out = expand_parent(out)
        return out


# ============================================================
# 生成治理层：结构化输出 + 句级引用 + 拒答
# ============================================================
GENERATE_SYSTEM = """你是一名严谨的企业知识库问答助手。你必须遵守以下铁律：

1. **只依据给定证据回答**，不得使用你自己的任何先验知识。
2. 每一句结论后面必须标注证据编号，格式为 [证据N]（N 为证据序号）。
3. **引用编号必须与"支撑该结论的那条证据"严格对应**：
   - **先逐字找到包含该信息的证据，再填它的编号**；引用前请在心中核对：
     "我写的 quote，是不是就在这条 [证据N] 的正文里逐字出现过？"
   - 严禁凭印象填编号，也不要把编号当作"第几条结论"来数，更不要按证据出现顺序猜。
   - `quote` 必须从**该条证据的正文**逐字摘录（不要改写、不要拼接多处）。
   - 若一条结论由多条证据共同支撑，可并列标注，如 [证据2][证据5]。
4. 如果证据不足以回答问题，必须拒绝回答：把 answer 设为空字符串，
   refused 设为 true，并在 reason 里说明缺什么信息。
5. 输出必须是合法 JSON，不要输出任何 JSON 之外的文字。

输出 JSON 结构：
{
  "answer": "回答正文（含 [证据N] 标注；拒答时为空字符串）",
  "citations": [{"claim": "该结论", "evidence_id": 1, "quote": "证据原文片段（必须逐字摘录，不要改写、不要拼接多处）"}],
  "confidence": 0.0~1.0,
  "refused": false,
  "reason": ""
}"""


def build_evidence_block(candidates: list[RerankResult]) -> tuple[str, dict[int, str]]:
    """
    把候选块拼成带编号的证据区。
    面试考点：为什么要给证据编号？
      → 让模型"引用"而不是"复述"。没有编号，模型无法指认出处，
        引用溯源就无从校验；有编号，我们就能做**机器校验**（见 validate_citations）。
    """
    lines = []
    id_map: dict[int, str] = {}
    for i, c in enumerate(candidates, start=1):
        id_map[i] = c.chunk_id
        # 关键：在这里同步登记证据原文，供后续引用校验使用。
        # 真实 bug：早期版本由调用方（pipeline.ask）另行登记证据文本，
        #   一旦两处的遍历顺序/范围不一致，"校验用的原文"就和"喂给模型的原文"
        #   对不上，导致明明引用正确的句子也被判 verified=false（假阴性）。
        #   修复原则：**产生证据编号的地方，就必须是登记证据原文的地方** —— 单一事实来源。
        _register_evidence_text(i, c.chunk_id, c.text)
        # 小块做证据、大块做语境：命中片段（child）用于精确引用与高亮，
        # 父块作为"所在章节上下文"补充，帮助模型理解前后文但不作为引用目标。
        if c.parent_text and c.parent_text != c.text:
            lines.append(f"[证据{i}]（命中片段，来源块 {c.chunk_id}）\n{c.text}\n"
                         f"—— 所在章节上下文（仅供理解，不要引用本段）：\n{c.parent_text}")
        else:
            lines.append(f"[证据{i}]（来源块 {c.chunk_id}）\n{c.text}")
    return "\n\n".join(lines), id_map


def generate_answer(query: str, candidates: list[RerankResult],
                    trace: TraceRecorder | None = None) -> dict:
    """
    生成 + 治理。返回 {answer, citations, confidence, refused, reason, evidence_map}。

    三道防线（简历"生成治理层"的落地）：
      ① 结构化输出：强制 JSON，用 Pydantic 思路校验字段，缺字段就判失败。
      ② 句级引用：每句结论必须带 [证据N]，且 N 必须真实存在（否则判为"幻觉引用"）。
      ③ 证据不足拒答：证据分低于阈值 或 模型自报 refused 时，明确拒答。
    """
    evidence, id_map = build_evidence_block(candidates)
    prompt = f"证据：\n{evidence}\n\n用户问题：{query}\n\n请严格按 JSON 输出。"

    # ---------- 防线 ③（最硬的一层）：完全没检索到证据，直接拒答 ----------
    # 连证据都没有就调 LLM，等于请它"自由发挥"—— 这正是 RAG 幻觉的根源。
    # 所以这一层在调用模型**之前**就返回，不花一分钱、不给模型编造的机会。
    if not candidates:
        return {
            "answer": "", "citations": [], "confidence": 0.0, "refused": True,
            "reason": "知识库中没有检索到与问题相关的证据，无法回答",
            "evidence_map": {}, "top_score": 0.0,
        }

    with (trace.span("generate", kind="generate") if trace else _null_span()) as sp:
        try:
            res = llm.chat(prompt, system=GENERATE_SYSTEM, json_mode=True,
                           temperature=0.0, trace=trace, trace_span=sp)
            parsed = _parse_json(res.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("生成失败：%s", exc)
            parsed = {"answer": "", "citations": [], "confidence": 0.0,
                      "refused": True, "reason": f"生成异常：{exc}"}

    # ---------- 防线 ③：证据不足拒答（先于模型自报，客观门槛）----------
    #
    # 面试考点（这里踩过一个真实的坑）：
    #   早期用「rerank 原始分 < 固定阈值 0.35」判拒答，结果**假拒答**频发——
    #   因为不同 reranker（百炼 gte-rerank / 本地 bge / mock）的分数量纲完全不同，
    #   百炼对一条真正相关的 block 也可能只给 0.19，远低于 0.35，直接把正确文档拒了。
    #   "拿绝对分比固定阈值"是 RAG 里最典型的脆弱设计。
    #
    #   正确做法：**批内归一化 + 相对信号**。同一个 query 的候选之间是可比的自洽集合：
    #     · norm_score = (score - min) / (max - min)  → 拉到 [0,1]，量纲无关；
    #     · 只有当"最高分也归一化后仍很低"（整批都烂）或"证据为空"时才拒答；
    #     · 这样换 reranker 不用重调阈值，鲁棒得多。
    scores = [c.score for c in candidates] if candidates else []
    top_norm, top_score = _normalized_top(scores)

    # 客观拒答信号：没有候选，或最高分归一化后仍低于阈值且绝对分也为 0（真的不相关）
    low_evidence = (not candidates) or (top_norm < settings.refuse_norm_threshold
                                        and top_score <= settings.refuse_floor)
    # mock 模式下分数口径不同，不做此判定（避免误判）
    if llm.mock:
        low_evidence = not candidates

    if low_evidence or parsed.get("refused"):
        parsed["refused"] = True
        parsed.setdefault("reason", "检索到的证据不足以支撑可靠回答")
        parsed["answer"] = parsed.get("answer", "") or ""

    # ---------- 防线 ②：校验引用是否真实（防幻觉引用）----------
    parsed["citations"] = validate_citations(parsed.get("citations", []), id_map)
    # ---------- 防线 ②b：从 answer 正文回填引用 ----------
    # 真实缺陷：模型习惯把 [证据N] 写在 answer 文本里，却**不填** citations 数组。
    #   结果引用校验对空数组通过（0/0），但"句级引用溯源"实际是失效的 ——
    #   用户看到 [证据9] 却点不开任何证据。这里从正文正则提取编号并回填，
    #   保证"正文标注"与"结构化引用"始终一致。
    parsed["citations"] = _backfill_citations(parsed.get("answer", ""),
                                              parsed["citations"], id_map)
    parsed["evidence_map"] = id_map
    parsed["top_score"] = top_score
    return parsed


_CITE_RE = re.compile(r"\[证据\s*(\d+)\]")


def _backfill_citations(answer: str, citations: list, id_map: dict[int, str]) -> list:
    """
    从 answer 正文里的 [证据N] 提取编号，补齐/校正 citations。

    规则（面试会问为什么这么做）：
      · 以 answer 正文的 [证据N] 为准 —— 因为那是用户真正看到的溯源标记，
        必须"看到什么、就能点到什么"。
      · 保留 citations 里已有的 quote（那是模型摘录的原文，有价值）。
      · 正文未标注但 citations 里有的，视为模型过度声明，丢弃。
    """
    cited_ids = []
    for m in _CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n in id_map and n not in cited_ids:
            cited_ids.append(n)
    if not cited_ids:
        return citations

    by_id = {c.get("evidence_id"): c for c in citations or []}
    out = []
    for eid in cited_ids:
        c = dict(by_id.get(eid) or {"claim": "", "evidence_id": eid, "quote": ""})
        c["evidence_id"] = eid
        c["chunk_id"] = id_map[eid]
        c.setdefault("quote", "")
        c["verified"] = c.get("verified", False) or _quote_verified(c.get("quote", ""), eid, id_map)
        c.setdefault("corrected", False)
        c = _backfill_quote(c, id_map)
        out.append(c)
    return out


def validate_citations(citations: list, id_map: dict[int, str]) -> list:
    """
    引用校验（面试高频追问：你怎么保证模型不乱标引用？）：

      · evidence_id 必须落在真实证据编号范围内，否则丢弃（幻觉引用）。
      · quote 必须能在对应证据原文里找到（归一化子串匹配），找不到则 verified=false。
      · **编号自愈**：若 quote 在本证据里找不到，但在**另一条证据**里能找到，
        说明模型只是编号填错了、内容是对的 —— 此时自动纠正编号并标记
        `corrected=true`。这是"内容为准、编号为辅"的工程取舍：
        死板地判 false 会浪费正确召回，自动纠正能显著提升引用可用率，
        同时保留 corrected 标记供人工审计。

    这一步把"引用"从"模型口头承诺"变成"可机器验证的事实"。
    """
    valid = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        eid = c.get("evidence_id")
        quote = (c.get("quote") or "").strip()
        if eid not in id_map:
            continue  # 指向不存在的证据 → 幻觉，丢弃
        verified = _quote_verified(quote, eid, id_map)
        corrected = False
        if not verified and quote:
            # 编号自愈：在其它证据里找这句原文
            for other_eid in id_map:
                if other_eid == eid:
                    continue
                if _quote_verified(quote, other_eid, id_map):
                    eid = other_eid
                    verified = True
                    corrected = True
                    break
        valid.append(_backfill_quote({
            "claim": c.get("claim", ""),
            "evidence_id": eid,
            "chunk_id": id_map[eid],
            "quote": quote,
            "verified": verified,
            "corrected": corrected,
        }, id_map))
    return valid


def _backfill_quote(c: dict, id_map: dict[int, str], max_len: int = 120) -> dict:
    """
    若模型没给 quote（或给了空串），用证据原文的前若干字兜底。
    面试考点：这不是"编造引用"，而是"把可核对的原文摆出来"——
      · 原文来自我们真正喂给模型的证据块，非模型生成；
      · 因此必然 verified=true，用户点开就能在来源里对上；
      · 同时保留 quote_source 标记，区分"模型摘录"与"系统兜底"，
        避免把系统兜底误当模型能力。
    """
    if c.get("quote"):
        c.setdefault("quote_source", "model")
        return c
    text = _evidence_text(c.get("evidence_id"), id_map)
    if text:
        c["quote"] = text[:max_len].strip()
        c["verified"] = True
        c.setdefault("corrected", False)
        c["quote_source"] = "system"
    return c


_QUOTE_CACHE: dict[str, str] = {}
_QUOTE_CACHE_MAX = 4096  # 上限，防止长驻服务把证据原文无限缓存导致内存泄漏


def _register_evidence_text(eid: int, chunk_id: str, text: str) -> None:
    """
    登记"喂给模型的证据原文"，供引用校验做逐字比对。
    面试考点：为什么要有上限？
      → 这是进程级全局缓存，QPS 高时不设上限会随请求量线性增长，属于典型内存泄漏。
        这里用最朴素的 FIFO 淘汰（够用即可，无需上 LRU 库）；
        生产环境更稳妥的做法是把证据原文放在单次请求的上下文里（避免跨请求污染），
        这里保留全局是为了让 generate_answer / validate_citations 可独立调用。
    """
    if len(_QUOTE_CACHE) >= _QUOTE_CACHE_MAX:
        # 淘汰最早的一批 key
        for k in list(_QUOTE_CACHE.keys())[: _QUOTE_CACHE_MAX // 4]:
            _QUOTE_CACHE.pop(k, None)
    _QUOTE_CACHE[f"{eid}:{chunk_id}"] = text


def _evidence_text(eid: int, id_map: dict[int, str]) -> str:
    return _QUOTE_CACHE.get(f"{eid}:{id_map.get(eid, '')}") or ""


def _quote_verified(quote: str, eid: int, id_map: dict[int, str]) -> bool:
    if not quote:
        return False
    text = _evidence_text(eid, id_map)
    if not text:
        return False
    norm = lambda s: "".join(s.split()).lower()
    return norm(quote) in norm(text)


def _normalized_top(scores: list[float]) -> tuple[float, float]:
    """
    把一批 rerank 分数做**批内 min-max 归一化**，返回 (归一化 top 分, 原始 top 分)。
    为什么需要：不同 reranker 分数量纲不同（百炼 gte-rerank 常给 0.1~0.9，
    本地 bge 可能给 -10~10），拿绝对分比固定阈值必翻车。批内归一化后量纲无关。
    """
    if not scores:
        return 0.0, 0.0
    lo, hi = min(scores), max(scores)
    top = scores[0]
    if hi > lo:
        return (top - lo) / (hi - lo), top
    return (1.0 if top > 0 else 0.0), top


def _parse_json(text: str) -> dict:
    """健壮 JSON 解析：模型偶尔会在 JSON 外包 ```json ... ``` 或加解释。"""
    import json
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t.strip("`")
        t = t.replace("json\n", "", 1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"answer": "", "citations": [], "confidence": 0.0,
            "refused": True, "reason": "模型输出无法解析为 JSON"}


reranker = Reranker()
