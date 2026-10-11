"""
知枢 · 文档解析层（多模态）
============================================================
简历原文：「MinerU + PaddleOCR 多模态流水线处理扫描件 / 复杂表格 / 双栏排版」

本模块把这个能力做成**可插拔的解析器链**：

  PDF  ──▶ ① 文本层检测（PyMuPDF）
            ├─ 有文本层 ──▶ 直接抽取（快、准、无 OCR 误差）
            └─ 无文本层（扫描件）──▶ ② 位图化 + OCR
                                        ├─ MinerU（首选，保留版面/表格结构）
                                        └─ PaddleOCR（MinerU 不可用时的降级）

面试考点 1：为什么不直接全上 OCR？
  → OCR 有误差（尤其数字、公式），有文本层的 PDF 直接抽取是"零误差"。
    先用文本层、只在必要时 OCR，是准确率与成本的双重最优。

面试考点 2：为什么分"扫描件 / 表格 / 双栏"分别处理？
  → 双栏 PDF 如果按"从左到右逐行"读，会把左右两栏的文字交错串起来，
    语义全乱。必须做**版面分析**（检测分栏）后再按栏阅读。
    这是通用 PDF 抽取器最常见的翻车点，也是"复杂文档解析"的价值所在。

工程取舍（诚实标注）：
  · MinerU / PaddleOCR 是重依赖（含深度学习模型，体积大、要 GPU 更佳）。
    本项目把它们封在可选适配器里：装了就启用（settings 自动探测），
    没装则降级到 PyMuPDF 文本抽取 + 简单版面启发式。
  · 降级的解析质量**明显低于** MinerU/OCR，README 中已明确标注，
    绝不把降级路径的效果当成 MinerU 的效果对外宣称。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("zhishu.ingest")


@dataclass
class ParsedBlock:
    """解析产出的一个语义块（段落 / 表格 / 标题），保留版面信息。"""
    text: str
    block_type: str = "paragraph"   # paragraph | table | title | list
    page: int = 1
    column: int = 0                 # 双栏时：0=左栏 1=右栏
    bbox: tuple[float, float, float, float] | None = None
    level: int = 0                  # 标题层级


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    source_path: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    page_count: int = 0
    parser: str = "unknown"         # 实际生效的解析器，用于"诚实标注"
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text.strip())


# ============================================================
# 解析器能力探测：装了就用，没装就降级（并如实记录）
# ============================================================
def _has_mineru() -> bool:
    try:
        import magic_pdf  # MinerU 的包名是 magic-pdf
        return True
    except Exception:  # noqa: BLE001
        return False


def _has_paddleocr() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _has_pymupdf() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def detect_parsers() -> dict[str, bool]:
    return {"mineru": _has_mineru(), "paddleocr": _has_paddleocr(), "pymupdf": _has_pymupdf()}


# ============================================================
# 解析器 1：PyMuPDF 文本层（基线，永远可用）
# ============================================================
class PyMuPDFParser:
    """
    基线解析器：直接抽文本层，并做**双栏检测**。

    双栏检测思路（面试会问）：
      → 取页面所有文本块的 x 中心，做一维聚类；若明显分成两簇且中间有空白带，
        判定为双栏，然后把块按 (栏, y) 排序，保证阅读顺序正确。
        比"逐行 y 排序"多这一步，就能避免左右栏文字交错。
    """

    name = "pymupdf"

    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        import fitz

        doc = fitz.open(str(path))
        out = ParsedDoc(doc_id=doc_id, title=_guess_title(doc), source_path=str(path),
                        page_count=doc.page_count, parser=self.name)
        for pno, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
            blocks = [b for b in blocks if b[4].strip()]
            two_col, split_x = _detect_two_column(blocks, page.rect.width)
            if two_col:
                out.warnings.append(f"第{pno}页检测为双栏排版，已按栏重排阅读顺序")
            # 排序：先栏、后纵向
            def sort_key(b):
                cx = (b[0] + b[2]) / 2
                col = 0 if cx < split_x else 1
                return (col, round(b[1], 1))
            for b in sorted(blocks, key=sort_key):
                cx = (b[0] + b[2]) / 2
                col = 0 if cx < split_x else 1
                txt = _clean_text(b[4])
                if not txt:
                    continue
                btype = _classify_block(txt, b, page.rect.width)
                out.blocks.append(ParsedBlock(text=txt, block_type=btype, page=pno,
                                              column=col, bbox=(b[0], b[1], b[2], b[3])))
        doc.close()
        return out


def _detect_two_column(blocks: list, page_width: float) -> tuple[bool, float]:
    """基于 x 中心分布判断是否双栏；返回 (是否双栏, 分界线x)。"""
    if len(blocks) < 6:
        return False, page_width / 2
    centers = sorted((b[0] + b[2]) / 2 for b in blocks)
    mid = page_width / 2
    left = [c for c in centers if c < mid]
    right = [c for c in centers if c >= mid]
    # 双栏特征：两栏各占一定块数，且中间有空白带（无块跨越中线）
    if len(left) >= 3 and len(right) >= 3:
        crossing = sum(1 for b in blocks if b[0] < mid - 20 and b[2] > mid + 20)
        if crossing <= max(1, len(blocks) * 0.15):
            return True, mid
    return False, mid


def _classify_block(text: str, bbox: tuple, page_width: float) -> str:
    """粗分类：标题 / 表格 / 列表 / 段落。"""
    t = text.strip()
    if re.match(r"^(第[一二三四五六七八九十]+[章节条]|[0-9]+(\.[0-9]+)*[、.．])", t) or (
        len(t) < 40 and (t.endswith("：") or _is_short_heading(t))
    ):
        return "title"
    if re.match(r"^[\-\u2022\*]\s|^[（(]?[0-9]+[)）]\s", t):
        return "list"
    # 表格启发式：同一行有多个 2+ 空格分隔的列
    if len(re.findall(r"\S+\s{2,}\S+", t)) >= 2:
        return "table"
    return "paragraph"


def _is_short_heading(t: str) -> bool:
    return len(t) <= 20 and not re.search(r"[。；，,]$", t)


def _clean_text(t: str) -> str:
    t = t.replace("\u3000", " ").replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _guess_title(doc: Any) -> str:
    meta = doc.metadata or {}
    if meta.get("title"):
        return str(meta["title"]).strip()
    # 退而求其次：取第一页最长的一段短文本
    for page in doc:
        for b in page.get_text("blocks"):
            txt = b[4].strip()
            if 4 < len(txt) < 40 and "\n" not in txt:
                return txt
        break
    return "未命名文档"


class TextParser:
    """
    纯文本 / Markdown 解析器。

    面试考点：为什么 .md 不能送进 PDF 解析器？
      → 不同文件类型有各自的解析路径。把 md 当 PDF 喂给 PyMuPDF 会直接报
        "Failed to open file" —— 这类"类型不匹配"的 bug 只有在**真实数据**
        上跑才会暴露，纯单元测试用 mock 是发现不了的。
    做法：按扩展名分发到对应解析器，markdown 保留标题层级（# ~ ######）。
    """

    name = "markdown"

    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        out = ParsedDoc(doc_id=doc_id, title=path.stem, source_path=str(path),
                        page_count=1, parser=self.name)
        if not raw.strip():
            out.warnings.append("文件为空或无法解码")
            return out

        # Markdown 才有 # 标题；纯 txt 无层级
        is_md = path.suffix.lower() in (".md", ".markdown")
        para_buf: list[str] = []

        def flush_para():
            if para_buf:
                text = " ".join(para_buf).strip()
                if text:
                    out.blocks.append(ParsedBlock(text=text, block_type="paragraph"))
                para_buf.clear()

        for line in raw.splitlines():
            s = line.rstrip()
            if not s.strip():
                flush_para()
                continue
            if is_md and re.match(r"^#{1,6}\s", s):
                flush_para()
                level = len(s) - len(s.lstrip("#"))
                out.blocks.append(ParsedBlock(text=s.lstrip("# ").strip(),
                                              block_type="title", level=level))
            elif re.match(r"^\s*[-*\u2022]\s+", s) or re.match(r"^\s*\d+[.)]\s+", s):
                flush_para()
                out.blocks.append(ParsedBlock(text=re.sub(r"^\s*([-*\u2022]|\d+[.)])\s+", "", s),
                                              block_type="list"))
            else:
                para_buf.append(s.strip())
        flush_para()
        return out


# ============================================================
# 解析器 2：MinerU（首选，保留版面与表格结构）
# ============================================================
class MinerUParser:
    """
    MinerU（magic-pdf）适配器：能把 PDF 转成结构化 markdown，
    对表格、公式、多栏的支持显著优于纯文本抽取。

    面试诚实点：MinerU 体积大（模型数 GB），本项目把它做成**可选**。
    未安装时本解析器不可用，会被自动跳过（见 parse_document 的链式降级）。
    """
    name = "mineru"

    def available(self) -> bool:
        return _has_mineru()

    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        out_dir = Path(path).parent / "_mineru_out"
        out_dir.mkdir(exist_ok=True)
        writer = FileBasedDataWriter(str(out_dir))
        ds = PymuDocDataset(Path(path).read_bytes())
        method = SupportedPdfParseMethod.OCR if ds.classify() == "ocr" else SupportedPdfParseMethod.TXT
        infer = doc_analyze(ds, ocr=True) if method == SupportedPdfParseMethod.OCR else doc_analyze(ds)
        infer.pipe_txt_mode("auto").pipe_mk_markdown(f"{doc_id}", writer, [])

        md_file = out_dir / f"{doc_id}.md"
        md = md_file.read_text(encoding="utf-8", errors="ignore") if md_file.exists() else ""

        out = ParsedDoc(doc_id=doc_id, title=md.splitlines()[0][:60] if md else doc_id,
                        source_path=str(path), parser="mineru")
        for line in md.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                level = len(s) - len(s.lstrip("#"))
                out.blocks.append(ParsedBlock(text=s.lstrip("# ").strip(), block_type="title", level=level))
            elif s.startswith("|") and s.endswith("|"):
                out.blocks.append(ParsedBlock(text=s, block_type="table"))
            elif re.match(r"^[\-\*\u2022]\s", s):
                out.blocks.append(ParsedBlock(text=s, block_type="list"))
            else:
                out.blocks.append(ParsedBlock(text=s, block_type="paragraph"))
        return out


# ============================================================
# 解析器 3：PaddleOCR（扫描件兜底）
# ============================================================
class PaddleOCRParser:
    """
    扫描件 OCR 兜底：把 PDF 每页位图化后用 PaddleOCR 识别。
    OCR 天然会有字符误差（尤其 0/O、1/l、小数点），
    所以本解析器会在 warnings 里提示"OCR 结果建议人工抽检"。
    """
    name = "paddleocr"

    def available(self) -> bool:
        return _has_paddleocr() and _has_pymupdf()

    def parse(self, path: Path, doc_id: str) -> ParsedDoc:
        import fitz
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        doc = fitz.open(str(path))
        out = ParsedDoc(doc_id=doc_id, title=path.stem, source_path=str(path),
                        page_count=doc.page_count, parser="paddleocr")
        out.warnings.append("本次为 OCR 解析，字符级误差不可避免，重要数字建议人工抽检")
        for pno, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            img_path = Path(path).parent / f"_ocr_p{pno}.png"
            pix.save(str(img_path))
            result = ocr.ocr(str(img_path), cls=True)
            lines = []
            for page_res in (result or []):
                for item in (page_res or []):
                    lines.append(item[1][0])
            out.blocks.append(ParsedBlock(text="\n".join(lines), page=pno, block_type="paragraph"))
            img_path.unlink(missing_ok=True)
        doc.close()
        return out


# ============================================================
# 链式降级：按优先级挑选可用解析器
# ============================================================
def _has_text_layer(path: Path) -> bool:
    """判断 PDF 是否有可抽取文本层。扫描件通常没有（或极少）。"""
    if not _has_pymupdf():
        return False
    import fitz
    try:
        doc = fitz.open(str(path))
        total = 0
        for page in doc:
            total += len(page.get_text().strip())
            if total > 200:
                doc.close()
                return True
        doc.close()
    except Exception:  # noqa: BLE001
        return False
    return False


def parse_document(path: str | Path, doc_id: str | None = None,
                   prefer: str | None = None) -> ParsedDoc:
    """
    解析调度：根据文件类型与可用组件选解析器。

    优先级（有文本层的 PDF）：
        MinerU  →  PyMuPDF
    优先级（扫描件 / 无文本层）：
        MinerU  →  PaddleOCR  →  PyMuPDF（抽不到文本，产空并告警）

    `prefer` 可强制指定，便于测试与对比不同解析器的效果。
    这是"同一份文档用不同解析器效果差多少"这类面试追问的抓手。
    """
    path = Path(path)
    if doc_id is None:
        import hashlib
        doc_id = "doc_" + hashlib.md5(str(path).encode()).hexdigest()[:10]

    if prefer:
        forced = {"mineru": MinerUParser(), "pymupdf": PyMuPDFParser(),
                  "paddleocr": PaddleOCRParser(), "markdown": TextParser()}.get(prefer)
        if forced is None:
            raise ValueError(f"未知解析器: {prefer}")
        return forced.parse(path, doc_id)

    # ---------- 按文件类型分发（不同类型走不同解析链）----------
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown", ".csv"):
        # 纯文本类不需要 PDF/OCR 解析，直接用文本解析器
        return TextParser().parse(path, doc_id)

    if suffix != ".pdf":
        logger.warning("未知文件类型 %s，尝试按纯文本解析", suffix)
        return TextParser().parse(path, doc_id)

    # ---------- PDF：根据有无文本层选择链路 ----------
    has_text = _has_text_layer(path)
    chain: list[Any] = []
    if _has_mineru():
        chain.append(MinerUParser())
    if not has_text and _has_paddleocr():
        chain.append(PaddleOCRParser())
    if _has_pymupdf():
        chain.append(PyMuPDFParser())

    if not chain:
        raise RuntimeError("没有任何可用解析器：请安装 pymupdf（基线）或 mineru / paddleocr")

    last_err: Exception | None = None
    for parser in chain:
        try:
            doc = parser.parse(path, doc_id)
            if doc.blocks:
                logger.info("解析 %s 使用 %s，产出 %d 块", path.name, doc.parser, len(doc.blocks))
                return doc
            logger.warning("%s 解析 0 块，尝试下一个解析器", parser.name)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("%s 解析失败：%s，降级到下一个", parser.name, exc)

    # 全部失败：返回带告警的空文档，不抛异常（上层可决策）
    doc = ParsedDoc(doc_id=doc_id, title=path.stem, source_path=str(path), parser="none")
    doc.warnings.append(f"所有解析器均失败：{last_err}")
    return doc
