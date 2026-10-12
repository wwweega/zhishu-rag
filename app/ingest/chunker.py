"""
知枢 · 分块层（语义分块 + 父子块 small-to-big）
============================================================
简历原文：「语义分块 + 父子块 small-to-big（检索用小块、生成喂父块）
           + LLM 生成块级标题与摘要」

为什么需要父子块：
  这是 RAG 里最经典的矛盾 ——
    · 块太大：向量被"稀释"，一篇文章混成一个向量，召回不准；
    · 块太小：检索准了，但喂给 LLM 的上下文太碎，答不全、易断章取义。
  父子块的解法是把两者解耦：
    · 用 **child（小块）** 建向量做检索 —— 精准命中；
    · 命中后回溯它的 **parent（大块）** 喂给 LLM —— 上下文完整。
  一句话：**小块负责"找到"，大块负责"看懂"**。

三层实现：
  L1 语义分块：按结构边界（标题/段落/句子）切，超长再按句滑窗，
              而不是粗暴按 512 字符硬切（硬切会把一句话劈两半）。
  L2 父子块：parent ≈ 2~4 个 child 合并，child 带 parent_id。
  L3 块级摘要与标题：LLM 为每个 parent 生成标题与摘要，
              摘要作为"块的门牌号"参与检索，提升可解释性。
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.ingest.parsers import ParsedBlock, ParsedDoc

logger = logging.getLogger("zhishu.chunk")


@dataclass
class Chunk:
    """入库单元。child 与 parent 用同一结构，靠 is_parent 区分。"""
    chunk_id: str
    doc_id: str
    text: str
    parent_id: str | None = None
    ordinal: int = 0
    token_est: int = 0
    is_parent: bool = False
    title: str = ""
    summary: str = ""
    meta: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """内容指纹：用于幂等去重（同一段文本重复入库时直接跳过）。"""
        return hashlib.md5(self.text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """
    粗估 token 数。中文约 1 token ≈ 1.5 汉字；英文约 1 token ≈ 4 字符。
    设计要点：为什么要估而不是精确算？
      → 分块阶段还没调 LLM，没有精确 usage。估算用于"块大小控制"足够，
        真正计费以 API 返回的 usage 为准（见 observability.record_llm）。
    """
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


def split_sentences(text: str) -> list[str]:
    """中文友好的句子切分：在句末标点后断开，但避免切断小数、缩写、列举。"""
    text = re.sub(r"\s+", " ", text)
    # 在 [。！？；] 或 英文句点(后跟空格/结尾) 处切
    parts = re.split(r"(?<=[。！？；!?])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])", text)
    return [p.strip() for p in parts if p and p.strip()]


def semantic_chunk_blocks(blocks: list[ParsedBlock], target_size: int,
                          overlap: int) -> list[Chunk]:
    """
    L1 语义分块：
      · 标题开启新块（标题天然是语义边界）
      · 表格整体成块（跨行切开表格 = 破坏语义）
      · 正文按句子累加，超过 target_size 就收束；相邻块保留 overlap 句做重叠，
        避免"答案正好卡在两块之间"。
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    title = ""

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = " ".join(buf).strip()
        if text:
            chunks.append(Chunk(chunk_id="", doc_id="", text=text,
                                token_est=estimate_tokens(text), title=title))
        # -------- 重叠窗口 --------
        # 设计要点：重叠有什么用？
        #   → 防止"答案正好横跨两块边界"导致两块都只命中半句。
        #     保留尾部若干句进入下一块，让上下文连续。
        # 注意（真实 bug）：早期实现里 `buf = keep if keep_tok < buf_tokens else []`，
        #   而 keep_tok 恒小于 buf_tokens，导致 buf 永远被保留、永不真正清空，
        #   最终整篇文档被合并成一个巨大的块。重叠量必须**显式上限**，
        #   且清空逻辑要无条件执行（这里用 keep 直接替换 buf）。
        keep: list[str] = []
        keep_tok = 0
        for s in reversed(buf):
            st = estimate_tokens(s)
            if keep_tok + st > overlap:
                break
            keep.insert(0, s)
            keep_tok += st
        buf = keep
        buf_tokens = keep_tok

    for b in blocks:
        if b.block_type == "title":
            flush()
            title = b.text
            buf = [b.text]
            buf_tokens = estimate_tokens(b.text)
            continue
        if b.block_type == "table":
            flush()
            chunks.append(Chunk(chunk_id="", doc_id="", text=b.text,
                                token_est=estimate_tokens(b.text), title=title))
            continue
        for sent in split_sentences(b.text):
            st = estimate_tokens(sent)
            if buf_tokens + st > target_size and buf:
                flush()
            buf.append(sent)
            buf_tokens += st
    flush()
    return chunks


def build_parent_child(semantic_chunks: list[Chunk], group_size: int = 3) -> tuple[list[Chunk], list[Chunk]]:
    """
    L2 父子块：把 group_size 个相邻语义块合并为一个 parent。

    追问：group_size 怎么定？
      → 它决定"生成时的上下文窗口"。太大 → 引入噪声、token 贵；
        太小 → 又退化成小块。经验值 2~4，本项目取 3。
        最终应以评测集上的 Faithfulness / Answer Relevancy 调优，
        而不是拍脑袋 —— 这也是本项目留了 group_size 参数的原因。
    """
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for i in range(0, len(semantic_chunks), group_size):
        group = semantic_chunks[i: i + group_size]
        if not group:
            continue
        pid = f"par_{hashlib.md5((group[0].text[:64] + str(i)).encode()).hexdigest()[:12]}"
        ptext = "\n".join(c.text for c in group)
        parents.append(Chunk(chunk_id=pid, doc_id="", text=ptext,
                             token_est=estimate_tokens(ptext), is_parent=True,
                             title=group[0].title, ordinal=i // group_size))
        # -------- 关键修复：把 group 里的每个语义块作为 child 收进 children --------
        # 之前这里漏了 children.extend(group)，导致"检索用小块"退化成
        # "检索用整篇父块"，父子块机制形同虚设。
        # 这类 bug 不会报错、不会崩，只会**静默地让召回质量变差** ——
        # 正是"跑通 ≠ 跑对"的典型，必须靠看数据（children 数是否为 0）发现。
        for j, c in enumerate(group):
            c.parent_id = pid
            c.ordinal = j
            children.append(c)
    return children, parents


def chunk_document(doc: ParsedDoc, group_size: int = 3) -> tuple[list[Chunk], list[Chunk]]:
    """
    文档 → (children, parents) 全流程。
    每个 chunk 都会带上文档级元数据（dept / doc_type），
    以便检索时做**元数据过滤**（权限 / 时效）。
    """
    sem = semantic_chunk_blocks(doc.blocks, settings.chunk_size, settings.chunk_overlap)
    children, parents = build_parent_child(sem, group_size)

    # -------- 先统一重写 parent 的 id，再用新 id 回填 child.parent_id --------
    # 真实 bug：build_parent_child 里用哈希生成了 parent_id，但这里又把
    # 父块的 chunk_id 覆写成 `par_{doc}_{idx}`，导致 child.parent_id 指向了
    # 一个**已不存在的旧 id** —— 结果"命中子块后回溯父块"永远查不到，
    # 父子块链路静默断裂（不报错，只是生成时拿不到完整上下文）。
    # 正确做法：先把父块 id 定死，再按 group 关系重新绑定 child。
    for idx, p in enumerate(parents):
        p.doc_id = doc.doc_id
        p.chunk_id = f"par_{doc.doc_id}_{idx:04d}"

    for idx, c in enumerate(children):
        c.doc_id = doc.doc_id
        c.chunk_id = f"chk_{doc.doc_id}_{idx:04d}"

    # 依据"child 在 sem 中的序号 // group_size"重新挂父块，与 parents 的顺序严格对齐
    for i, c in enumerate(sem):
        if i < len(children):
            children[i].parent_id = parents[i // group_size].chunk_id

    for is_parent, lst in ((False, children), (True, parents)):
        for c in lst:
            c.meta = {
                "doc_type": _infer_doc_type(doc),
                "dept": getattr(doc, "dept", "") or "",
                "effective_ts": getattr(doc, "effective_ts", 0) or 0,
            }
    logger.info("分块完成：%d child / %d parent（原文 %d 块）",
                len(children), len(parents), len(doc.blocks))
    return children, parents


def _infer_doc_type(doc: ParsedDoc) -> str:
    name = (doc.source_path or "").lower()
    if "制度" in name or "policy" in name:
        return "policy"
    if "手册" in name or "manual" in name:
        return "manual"
    if "合同" in name or "contract" in name:
        return "contract"
    return "general"
