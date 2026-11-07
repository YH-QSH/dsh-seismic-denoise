"""Word 报告生成（python-docx）。

契约（docs/接口约定.md §6）：
    build_report(job_dir, state) -> Path   # 05_报告/去噪过程报告.docx

报告固定 8 个正文大节 + 封面：
    封面 / 数据读取 / 来源匹配 / 去噪决策 / 尝试记录 /
    质量评价 / 置信度分析 / 结论与建议 / 文件清单

实现原则：
- 对 state 缺字段全部容错，缺什么写“—”，绝不因一个字段崩溃；
- 未通过门禁时封面、正文、结论三处醒目写“未通过行业门禁”，不粉饰；
- 干净数据（needs_denoise=False）生成“无需去噪”报告，仍保留读取与来源匹配；
- 图片按约定路径查找，宽 16cm；找不到时插入占位说明而不是抛异常；
- 正文宋体、标题黑体，表格统一 Table Grid。
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from ..model import SeismicDataset, to_jsonable
from ..utils import read_json
from . import figures as F
from . import package as pkg

# --------------------------------------------------------------------------
# 字体、颜色、常量
# --------------------------------------------------------------------------
BODY_FONT = "宋体"
HEAD_FONT = "黑体"
MONO_FONT = "Consolas"

C_DARK = RGBColor(0x1F, 0x38, 0x64)     # 深蓝：一级标题
C_MID = RGBColor(0x1F, 0x4E, 0x79)      # 中蓝：二级标题
C_BLUE = RGBColor(0x2E, 0x75, 0xB6)     # 浅蓝：三级标题
C_RED = RGBColor(0xC0, 0x00, 0x00)      # 失败 / 警告
C_GREEN = RGBColor(0x37, 0x56, 0x23)    # 通过
C_ORANGE = RGBColor(0xB4, 0x5F, 0x06)   # 注意
C_GRAY = RGBColor(0x7F, 0x7F, 0x7F)     # 次要信息
C_BLACK = RGBColor(0x1A, 0x1A, 0x1A)

PLUGIN_VERSION = "1.0"
IMAGE_WIDTH_CM = 16.0
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# 报告图片候选路径（按契约 §7 的成果目录；也兼容不同命名）
_IMAGE_CANDIDATES: dict[str, list[str]] = {
    "overview": ["02_读取与诊断/数据总览.png", "数据总览.png"],
    "header": ["02_读取与诊断/道头分析.png", "道头分析.png"],
    "spectrum": ["02_读取与诊断/频率谱.png", "02_读取与诊断/振幅谱.png",
                 "频率谱.png", "振幅谱.png"],
    "fk": ["02_读取与诊断/fk谱.png", "02_读取与诊断/F-K谱.png",
           "02_读取与诊断/FK谱.png", "fk谱.png"],
    "source": ["02_读取与诊断/来源匹配图.png", "来源匹配图.png"],
    "compare": ["03_去噪结果/去噪后剖面.png", "03_去噪结果/去噪前后对比.png",
                "去噪后剖面.png"],
    "diff": ["03_去噪结果/去噪差分.png", "去噪差分.png"],
    "metrics": ["04_质量评价/指标对比.png", "04_质量评价/质量指标.png",
                "指标对比.png"],
    "zoom": ["04_质量评价/局部放大对比.png", "局部放大对比.png"],
    "visual": ["04_质量评价/视觉对比.png", "视觉对比.png"],
}


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------
def _clean_text(value: Any) -> str:
    """去掉控制字符；表格单元内的换行压成空格。"""
    if value is None:
        return ""
    text = _CONTROL_RE.sub("", str(value))
    return text.replace("\r", " ").replace("\n", " ")


def _fmt(value: Any, digits: int = 2) -> str:
    """数值/布尔的统一显示，None 显示“—”。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        av = abs(value)
        if av != 0.0 and (av < 1e-3 or av >= 1e5):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    """把 0–1 的比例显示成百分比。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:.1f}%"


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _as_dict(obj: Any) -> dict:
    """dataclass / 带 to_info() 的对象 / dict 统一成 dict。"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_info"):
        try:
            info = obj.to_info()
            return info if isinstance(info, dict) else {}
        except Exception:
            return {}
    if is_dataclass(obj):
        try:
            return to_jsonable(asdict(obj))
        except Exception:
            return {}
    return {}


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """安全地按路径取值：dict 用 key，对象用属性。"""
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _load_state(state: Any) -> dict:
    """state 既可以是 dict，也可以是写出的 JSON 路径。"""
    if isinstance(state, dict):
        return state
    if isinstance(state, (str, Path)):
        data = read_json(state, default=None)
        return data if isinstance(data, dict) else {}
    return {}


def _plugin_version() -> str:
    """优先读插件根目录 package.json 的版本，读不到用 1.0。"""
    try:
        root = Path(__file__).resolve().parents[2]
        data = json.loads((root / "package.json").read_text(encoding="utf-8"))
        version = str(data.get("version") or "").strip()
        return version or PLUGIN_VERSION
    except Exception:
        return PLUGIN_VERSION


def _params_text(params: Any, limit: int = 180) -> str:
    """把参数字典压成一行“k=v, k=v”。"""
    if not isinstance(params, dict) or not params:
        return "—"
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}={list(value)}")
        else:
            parts.append(f"{key}={value}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _verdict(state: dict) -> tuple[str, str, str]:
    """返回 (标题, 说明, 类型)；类型：clean / ok / fail / unknown。"""
    source = _as_dict(state.get("source"))
    needs = source.get("needs_denoise") if source else None
    gate = _as_dict(state.get("gate"))
    if not gate:
        gate = _as_dict(_get(_as_dict(state.get("best")), "gate"))
    attempts = state.get("attempts") or []
    stage = state.get("stage") or ""

    if needs is False:
        return (
            "无需去噪",
            "数据质量良好，判定为无需去噪，未进入去噪与门禁流程。",
            "clean",
        )
    if gate.get("passed") is True:
        return ("达标", "去噪结果已通过行业门禁，可用于后续处理与解释。", "ok")
    if attempts or gate or stage in ("denoise", "done"):
        return (
            "未达标",
            "去噪结果未达到行业门禁要求，请勿直接用于定量解释。",
            "fail",
        )
    return ("未完成", "本次任务尚未产生可评价的去噪结果。", "unknown")


# --------------------------------------------------------------------------
# Word 底层格式工具
# --------------------------------------------------------------------------
def _set_run_font(
    run,
    name: str = BODY_FONT,
    size: float | None = None,
    bold: bool | None = None,
    color: RGBColor | None = None,
    italic: bool | None = None,
) -> None:
    """设置 run 的字体（含中文 eastAsia 与西文 ascii/hAnsi）。"""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def _set_style_font(
    style,
    name: str,
    size: float | None = None,
    bold: bool | None = None,
    color: RGBColor | None = None,
) -> None:
    """设置 Word 样式字体（标题黑体、正文宋体）。"""
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color


def _shade_cell(cell, fill: str) -> None:
    """单元格底色（十六进制 RGB，不带 #）。"""
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def _add_field(paragraph, instruction: str) -> None:
    """插入 Word 域（用于页码 PAGE / NUMPAGES）。"""
    run = paragraph.add_run()
    _set_run_font(run, BODY_FONT, 9, color=C_GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r = run._r
    for node in (begin, instr, separate, text, end):
        r.append(node)


def _setup_document(doc) -> None:
    """页面、正文/标题样式、页脚页码。"""
    # A4 纸，左右留 2.2cm，正文宽度 16.6cm，足够放 16cm 宽图
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.4)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    _set_style_font(normal, BODY_FONT, 10.5, color=C_BLACK)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.35

    heading_specs = {
        "Title": (HEAD_FONT, 28, C_DARK),
        "Heading 1": (HEAD_FONT, 16, C_DARK),
        "Heading 2": (HEAD_FONT, 13, C_MID),
        "Heading 3": (HEAD_FONT, 11.5, C_BLUE),
    }
    for name, (font_name, size, color) in heading_specs.items():
        try:
            style = doc.styles[name]
        except KeyError:
            continue
        _set_style_font(style, font_name, size, bold=True, color=color)
        style.paragraph_format.space_before = Pt(12 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(6)

    # 页脚：报告名 + 页码
    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("地震智能去噪报告 · 第 ")
    _set_run_font(run, BODY_FONT, 9, color=C_GRAY)
    _add_field(footer, "PAGE")
    run = footer.add_run(" 页 / 共 ")
    _set_run_font(run, BODY_FONT, 9, color=C_GRAY)
    _add_field(footer, "NUMPAGES")
    run = footer.add_run(" 页")
    _set_run_font(run, BODY_FONT, 9, color=C_GRAY)


# --------------------------------------------------------------------------
# 报告构建器
# --------------------------------------------------------------------------
class _ReportBuilder:
    def __init__(self, job_dir: str | Path, state: Any):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.state = _load_state(state)
        self.doc = Document()
        self.version = _plugin_version()
        self.fig_no = 0
        self.verdict, self.verdict_desc, self.verdict_kind = _verdict(self.state)
        _setup_document(self.doc)

        props = self.doc.core_properties
        props.title = "地震数据智能去噪报告"
        props.subject = "地震数据读取、来源匹配、去噪与质量评价"
        props.author = "地震智能去噪插件"
        props.comments = "由 engine.reporting.docx_report 自动生成"

    # ---- 便捷取数 ----
    @property
    def dataset(self) -> dict:
        return _as_dict(self.state.get("dataset"))

    @property
    def source(self) -> dict:
        return _as_dict(self.state.get("source"))

    def _best(self) -> dict:
        """最佳尝试：优先 state.best，否则取门禁得分最高的尝试。"""
        best = _as_dict(self.state.get("best"))
        if best:
            return best
        attempts = [a for a in (self.state.get("attempts") or []) if isinstance(a, dict)]
        if not attempts:
            return {}
        return max(attempts, key=lambda a: _num(_get(a, "gate", "score")) or -1.0)

    def _gate(self) -> dict:
        gate = _as_dict(self.state.get("gate"))
        return gate if gate else _as_dict(_get(self._best(), "gate"))

    def _metrics(self) -> dict:
        metrics = _as_dict(self.state.get("metrics"))
        if metrics:
            return metrics
        return _as_dict(_get(self._best(), "metrics"))

    def _needs_denoise(self) -> bool:
        value = self.source.get("needs_denoise")
        return True if value is None else bool(value)

    def _rel(self, path: Any) -> str:
        if not path:
            return "—"
        try:
            return Path(str(path)).resolve().relative_to(self.job_dir.resolve()).as_posix()
        except Exception:
            return str(path)

    # ---- 基础排版 ----
    def _p(
        self,
        text: Any,
        *,
        size: float = 10.5,
        bold: bool = False,
        color: RGBColor | None = None,
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        italic: bool = False,
        font: str = BODY_FONT,
        space_after: float = 4.0,
    ):
        p = self.doc.add_paragraph()
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(_clean_text(text))
        _set_run_font(run, font, size, bold, color, italic)
        return p

    def _h1(self, title: str) -> None:
        heading = self.doc.add_heading(title, level=1)
        for run in heading.runs:
            _set_run_font(run, HEAD_FONT, 16, True, C_DARK)

    def _h2(self, title: str) -> None:
        heading = self.doc.add_heading(title, level=2)
        for run in heading.runs:
            _set_run_font(run, HEAD_FONT, 13, True, C_MID)

    def _h3(self, title: str) -> None:
        heading = self.doc.add_heading(title, level=3)
        for run in heading.runs:
            _set_run_font(run, HEAD_FONT, 11.5, True, C_BLUE)

    def _bullets(self, items: Any, *, color: RGBColor | None = None, size: float = 10.0) -> None:
        for item in items or []:
            text = _clean_text(item)
            if not text:
                continue
            try:
                p = self.doc.add_paragraph(style="List Bullet")
                prefix = ""
            except KeyError:
                p = self.doc.add_paragraph()
                prefix = "• "
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(prefix + text)
            _set_run_font(run, BODY_FONT, size, color=color)

    def _cell(
        self,
        cell,
        text: Any,
        *,
        bold: bool = False,
        size: float = 9.0,
        shade: str | None = None,
        color: RGBColor | None = None,
        align=WD_ALIGN_PARAGRAPH.LEFT,
    ) -> None:
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(_clean_text(text))
        _set_run_font(run, BODY_FONT, size, bold, color)
        if shade:
            _shade_cell(cell, shade)

    def _table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        *,
        widths: list[float] | None = None,
        font_size: float = 9.0,
        header_fill: str = "D9E2F3",
    ) -> Any:
        """统一 Table Grid 表格：首行表头加粗加底色。"""
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        header_cells = table.rows[0].cells
        for idx, text in enumerate(headers):
            self._cell(header_cells[idx], text, bold=True, size=font_size,
                       shade=header_fill, align=WD_ALIGN_PARAGRAPH.CENTER)
        for row in rows:
            cells = table.add_row().cells
            for idx in range(len(headers)):
                value = row[idx] if idx < len(row) else ""
                self._cell(cells[idx], value, size=font_size)
        if widths:
            for idx, width in enumerate(widths):
                if idx >= len(table.columns):
                    break
                table.columns[idx].width = Cm(width)
                for cell in table.columns[idx].cells:
                    cell.width = Cm(width)
        # 表后留一点空隙，避免与下一段贴太紧
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return table

    def _kv_table(self, rows: list[tuple[str, Any]], headers=("项目", "内容")) -> None:
        self._table(list(headers), [[k, v] for k, v in rows], widths=[4.2, 12.2])

    def _figure(self, key: str, caption: str) -> None:
        """按关键名查找图片并插入（16cm 宽），找不到则写占位说明。"""
        path = self._find_image(key)
        if path is None:
            self._p(f"（未找到图片：{caption}）", size=9, color=C_GRAY, italic=True,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
            return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        try:
            p.add_run().add_picture(str(path), width=Cm(IMAGE_WIDTH_CM))
        except Exception as exc:
            self._p(f"（图片插入失败：{caption}；{exc}）", size=9, color=C_GRAY,
                    italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
            return
        self.fig_no += 1
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        run = cap.add_run(f"图 {self.fig_no}  {caption}")
        _set_run_font(run, BODY_FONT, 9, color=C_GRAY)

    def _find_image(self, key: str) -> Path | None:
        candidates = _IMAGE_CANDIDATES.get(key, [])
        for rel in candidates:
            path = self.job_dir / rel
            if path.is_file():
                return path
        # 兼容目录编号不同/放在别处：按文件名递归找
        for rel in candidates:
            name = Path(rel).name
            hits = sorted(self.job_dir.rglob(name))
            if hits:
                return hits[0]
        return None

    def _banner(self, text: str, kind: str) -> None:
        """整宽结论条：失败红色、通过绿色、说明灰色。"""
        fills = {"fail": "FBE4E4", "ok": "E2EFDA", "clean": "E2EFDA",
                 "unknown": "F2F2F2"}
        colors = {"fail": C_RED, "ok": C_GREEN, "clean": C_GREEN,
                  "unknown": C_GRAY}
        table = self.doc.add_table(rows=1, cols=1)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        cell.width = Cm(16.0)
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(_clean_text(text))
        _set_run_font(run, HEAD_FONT, 15, True, colors.get(kind, C_GRAY))
        _shade_cell(cell, fills.get(kind, "FFFFFF"))
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ---- 自动补图（缺图时尽力从缓存/state 里补生成） ----
    def _ensure_figures(self) -> None:
        diag = self.job_dir / "02_读取与诊断"
        den = self.job_dir / "03_去噪结果"
        quality = self.job_dir / "04_质量评价"
        for folder in (diag, den, quality):
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        ds = None
        cache = diag / "数据缓存.npz"
        if cache.exists():
            try:
                ds = SeismicDataset.load_cache(diag)
            except Exception:
                ds = None
        if ds is not None:
            dt = ds.dt if ds.dt else 0.002
            dz = ds.dz
            overview = diag / "数据总览.png"
            if not overview.exists():
                try:
                    F.fig_data_overview(ds.data, dt, dz, overview)
                except Exception:
                    pass
            spectrum = diag / "频率谱.png"
            if not spectrum.exists():
                try:
                    F.fig_spectrum(ds.data, dt, spectrum)
                except Exception:
                    pass
            fk = diag / "fk谱.png"
            if not fk.exists():
                try:
                    F.fig_fk(ds.data, dt, dz, fk)
                except Exception:
                    pass
            header = diag / "道头分析.png"
            if not header.exists():
                try:
                    F.fig_header_analysis(ds, header)
                except Exception:
                    pass
            source_image = diag / "来源匹配图.png"
            if self.source and not source_image.exists():
                try:
                    F.fig_source_assessment(ds, self.source, source_image)
                except Exception:
                    pass

        metrics = self._metrics()
        metrics_image = quality / "指标对比.png"
        if metrics and not metrics_image.exists():
            try:
                F.fig_metrics({"metrics": metrics, "gate": self._gate()},
                              metrics_image)
            except Exception:
                pass

    # ======================================================================
    # 各章节
    # ======================================================================
    def _cover(self) -> None:
        doc = self.doc
        for _ in range(2):
            doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("地震数据智能去噪报告")
        _set_run_font(run, HEAD_FONT, 30, True, C_DARK)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("—— 数据读取 · 来源匹配 · 去噪 · 质量评价 ——")
        _set_run_font(run, HEAD_FONT, 13, False, C_MID)

        doc.add_paragraph()
        input_raw = self.state.get("input") or self.dataset.get("path") or ""
        input_name = Path(str(input_raw)).name if input_raw else "（未记录输入文件）"
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(input_name)
        _set_run_font(run, HEAD_FONT, 17, True, C_BLACK)

        doc.add_paragraph()
        dataset = self.dataset
        rows = [
            ("数据文件", input_name),
            ("数据格式", f"{dataset.get('format', '—')}"
                       f"（{dataset.get('source_kind', 'unknown')}）"),
            ("数据规模", f"{_fmt(dataset.get('n_samples'))} 采样点 × "
                       f"{_fmt(dataset.get('n_traces'))} 道"),
            ("采样间隔", f"{_fmt(dataset.get('dt_ms'), 3)} ms"
                       f"；道间距 {_fmt(dataset.get('dz_m'), 3)} m"),
            ("处理阶段", self.state.get("stage", "—")),
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("插件版本", self.version),
            ("任务目录", self.job_dir.name),
        ]
        self._table(["项目", "内容"], [[k, v] for k, v in rows],
                    widths=[4.0, 12.0], font_size=10)
        doc.add_paragraph()

        banner = f"结论：{self.verdict}"
        if self.verdict_kind == "fail":
            banner += "（未通过行业门禁）"
        self._banner(banner, self.verdict_kind)
        self._p(self.verdict_desc, size=10.5,
                color=C_RED if self.verdict_kind == "fail" else C_GRAY,
                align=WD_ALIGN_PARAGRAPH.CENTER)
        if self.verdict_kind == "fail":
            self._p("警告：本结果未通过行业门禁，不得直接用于定量解释或成果提交。",
                    size=11, bold=True, color=C_RED,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
        doc.add_page_break()

    # ---- 一、数据读取 ----
    def _section_reading(self) -> None:
        self._h1("一、数据读取")
        dataset = self.dataset
        self._p("本报告的数据读取与诊断由插件自动完成。以下信息来自文件解析结果，"
                "读取置信度反映道头、采样与道数之间的一致性。")

        amplitude = _as_dict(dataset.get("amplitude"))
        rows = [
            ("数据路径", dataset.get("path") or self.state.get("input") or "—"),
            ("文件格式", dataset.get("format", "—")),
            ("来源类型", dataset.get("source_kind", "unknown")),
            ("采样点数", _fmt(dataset.get("n_samples"))),
            ("道数", _fmt(dataset.get("n_traces"))),
            ("采样间隔 dt", f"{_fmt(dataset.get('dt_ms'), 3)} ms"),
            ("道间距 dz", f"{_fmt(dataset.get('dz_m'), 3)} m"),
            ("记录长度", f"{_fmt(dataset.get('duration_s'), 3)} s"),
            ("振幅范围", f"{_fmt(amplitude.get('min'), 4)} ~ "
                       f"{_fmt(amplitude.get('max'), 4)}"),
            ("振幅均值 / 标准差", f"{_fmt(amplitude.get('mean'), 4)} / "
                              f"{_fmt(amplitude.get('std'), 4)}"),
            ("有限值比例", _pct(amplitude.get("finite_ratio"))),
            ("数据大小", self._human_bytes(dataset.get("data_bytes"))),
            ("读取置信度", _pct(dataset.get("confidence"))),
        ]
        self._kv_table(rows)

        warnings = dataset.get("warnings") or []
        if warnings:
            self._h2("读取警告")
            self._p(f"读取过程中记录到 {len(warnings)} 条警告，请结合原始数据核对：",
                    color=C_ORANGE)
            self._bullets(warnings, color=C_ORANGE)

        self._h2("道头分析")
        headers = dataset.get("headers") or {}
        if isinstance(headers, dict) and headers:
            header_rows = []
            ordered_keys = list(self._ordered_header_keys(headers))
            for key in ordered_keys[:16]:
                field = headers.get(key) or {}
                if not isinstance(field, dict):
                    continue
                stats = field.get("stats") or {}
                header_rows.append([
                    field.get("name") or key,
                    key,
                    field.get("byte", "—"),
                    field.get("kind", "—"),
                    _fmt(stats.get("min"), 3),
                    _fmt(stats.get("max"), 3),
                    _fmt(stats.get("mean"), 3),
                    _fmt(stats.get("unique")),
                ])
            if header_rows:
                self._table(
                    ["字段", "键", "起始字节", "类型", "最小值", "最大值",
                     "均值", "唯一值数"],
                    header_rows,
                    widths=[3.1, 2.3, 1.35, 1.8, 2.0, 2.0, 2.0, 1.5],
                    font_size=8.5,
                )
            else:
                self._p("道头字段均无有效数值。")
        else:
            self._p("本数据未解析出道头字段（可能是 NDArray/MAT 等无道头格式）。")
        self._figure("header", "关键道头字段随道号的变化")

        self._h2("数据总览与频谱")
        self._figure("overview", "数据总览：地震剖面、振幅分布与单道 RMS 能量")
        self._figure("spectrum", "平均振幅谱（阴影为 5–80 Hz 信号带）")
        self._figure("fk", "F-K 二维振幅谱")

        text_header = str(dataset.get("text_header") or "").strip()
        if text_header:
            self._h2("SEG-Y 文本头（节选）")
            self._p(text_header[:1200], size=8.5, font=MONO_FONT,
                    align=WD_ALIGN_PARAGRAPH.LEFT)

    @staticmethod
    def _human_bytes(value: Any) -> str:
        try:
            size = float(value)
        except (TypeError, ValueError):
            return "—"
        if size <= 0:
            return "—"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"

    @staticmethod
    def _ordered_header_keys(headers: dict):
        priority = ("trace_seq", "cdp", "ensemble", "inline", "crossline",
                    "offset", "shotpoint", "receiver_elev", "source_elev",
                    "source_depth", "receiver_depth", "cdp_x", "cdp_y")
        seen = set()
        for key in priority:
            if key in headers:
                seen.add(key)
                yield key
        for key in headers:
            if key not in seen:
                yield key

    # ---- 二、来源匹配 ----
    def _section_source(self) -> None:
        self._h1("二、来源匹配")
        source = self.source
        self._p("来源匹配根据道头、几何、频谱与噪声特征推断数据的领域、采集方式和"
                "噪声类型，用于决定是否需要去噪以及优先采用哪些方法。")
        if not source:
            self._p("本次任务没有来源匹配结果。", color=C_ORANGE)
            return

        rows = [
            ("数据领域", source.get("domain", "未知")),
            ("采集方式", source.get("acquisition", "未知")),
            ("观测系统", source.get("geometry_type", "未知")),
            ("匹配置信度", _pct(source.get("confidence"))),
            ("是否需要去噪", "是" if self._needs_denoise() else "否（无需去噪）"),
            ("几何信息", _params_text(source.get("geometry"), limit=300)),
            ("清洁度", _params_text(source.get("cleanliness"), limit=300)),
        ]
        self._kv_table(rows)

        noise_types = source.get("noise_types") or {}
        self._h2("噪声类型判定")
        if isinstance(noise_types, dict) and noise_types:
            advice_map = {
                "面波": "F-K 滤波 / 带通 / 中值滤波",
                "随机噪声": "f-x 反褶积 / 谱减 / CNN",
                "工频": "50/60 Hz 陷波及其谐波",
                "工频干扰": "50/60 Hz 陷波及其谐波",
                "散射绕射": "F-K / 局部 SVD / CNN",
                "折射": "F-K / 中值滤波",
                "异常振幅": "去尖峰 / 中值滤波",
                "未知": "保守带通 + CNN",
            }
            rows = []
            for name, prob in sorted(
                noise_types.items(),
                key=lambda kv: _num(kv[1]) if _num(kv[1]) is not None else -1,
                reverse=True,
            ):
                rows.append([name, _pct(prob), advice_map.get(str(name), "按方法规划器决定")])
            self._table(["噪声类型", "概率/强度", "建议处理方向"], rows,
                        widths=[4.2, 2.8, 9.6])
        else:
            self._p("未识别出明显的噪声类型。")

        self._h2("判定证据与结论")
        evidence = source.get("evidence") or []
        if evidence:
            self._p("判定证据：", bold=True)
            self._bullets(evidence)
        clean_reasons = source.get("clean_reasons") or []
        if clean_reasons:
            self._p("判定为干净数据的依据：", bold=True, color=C_GREEN)
            self._bullets(clean_reasons, color=C_GREEN)
        if source.get("notes"):
            self._p(f"补充说明：{source.get('notes')}")
        source_warnings = source.get("warnings") or []
        if source_warnings:
            self._p("来源匹配警告：", bold=True, color=C_ORANGE)
            self._bullets(source_warnings, color=C_ORANGE)

        recommended = source.get("recommended") or []
        if recommended:
            self._p(f"推荐方法：{'、'.join(str(x) for x in recommended)}")
        self._figure("source", "来源匹配与噪声类型概率")

    # ---- 三、去噪决策 ----
    def _section_plan(self) -> None:
        self._h1("三、去噪决策")
        source = self.source
        if not self._needs_denoise():
            self._p("数据被判定为“无需去噪”。按插件策略不进入去噪流程，"
                    "以下仅列出规划结果（如有）供参考。", bold=True, color=C_GREEN)
            clean_reasons = source.get("clean_reasons") or []
            if clean_reasons:
                self._bullets(clean_reasons, color=C_GREEN)
            self._p(f"推荐处理：{'、'.join(str(x) for x in (source.get('recommended') or ['保持原始数据']))}")

        plan = [p for p in (self.state.get("plan") or []) if isinstance(p, dict)]
        self._h2("候选方法")
        if plan:
            rows = []
            for idx, spec in enumerate(plan, start=1):
                rows.append([
                    idx,
                    spec.get("name") or spec.get("key") or "—",
                    spec.get("key", "—"),
                    "、".join(str(t) for t in (spec.get("targets") or [])) or "—",
                    _fmt(spec.get("strength"), 2),
                    _params_text(spec.get("params")),
                    spec.get("rationale", "—"),
                ])
            self._table(
                ["序号", "方法名", "键", "目标噪声", "强度", "参数", "选择理由"],
                rows,
                widths=[1.0, 2.8, 2.0, 3.4, 1.2, 3.0, 3.2],
                font_size=8.3,
            )
        else:
            self._p("本次没有生成候选方法（干净数据或流程尚未进入规划阶段）。")

        best = self._best()
        if best:
            self._h2("主方法选择")
            method = _as_dict(best.get("method"))
            rows = [
                ("方法名", method.get("name", "—")),
                ("方法键", method.get("key", "—")),
                ("方法来源", method.get("source", "—")),
                ("强度", _fmt(method.get("strength"), 2)),
                ("目标噪声", "、".join(str(t) for t in (method.get("targets") or [])) or "—"),
                ("选择理由", method.get("rationale") or "—"),
                ("参数", _params_text(method.get("params"), limit=400)),
            ]
            self._kv_table(rows)
            attempts = [a for a in (self.state.get("attempts") or []) if isinstance(a, dict)]
            passed = sum(1 for a in attempts if bool(_get(a, "gate", "passed")))
            self._p(f"共尝试 {len(attempts)} 种方法，其中 {passed} 种通过行业门禁；"
                    f"最终选择“{method.get('name', '—')}”作为主方法。")

    # ---- 四、尝试记录 ----
    def _metric_rows(self, metrics: dict) -> list[list[Any]]:
        specs = (
            ("snr_in_db", "去噪前盲信噪比", "dB", "越高越好，反映输入噪声水平"),
            ("snr_out_db", "去噪后盲信噪比", "dB", "门禁参考 ≥ 10 dB（盲估计不可靠时软检查）"),
            ("snr_gain_db", "信噪比增益", "dB", "硬门禁 ≥ 3.0 dB"),
            ("reference_gain_db", "相对参考增益", "dB", "有干净参考数据时计算"),
            ("noise_before_db", "去噪前噪声能量", "dB", "辅助指标"),
            ("noise_after_db", "去噪后噪声能量", "dB", "辅助指标"),
            ("noise_reduction_db", "噪声降低量", "dB", "越高说明压制越强，但需结合信号泄漏"),
            ("energy_ratio", "能量保持比", "—", "门禁 0.30 ~ 1.05，过低说明削掉太多能量"),
            ("signal_leakage", "信号泄漏", "—", "门禁 ≤ 0.40，越高越可能削到有效信号"),
            ("spectrum_corr", "5–80 Hz 频谱相关", "—", "门禁 ≥ 0.80"),
            ("coherence_residual", "残差相干性", "—", "辅助判断残差是否仍含构造"),
            ("clip_ratio", "削顶比例", "—", "门禁 ≤ 1e-3"),
            ("local_snr_gain_db", "局部 SNR 增益", "dB", "分窗统计，反映空间不均匀性"),
        )
        rows = []
        for key, name, unit, note in specs:
            value = metrics.get(key)
            if value is None:
                continue
            text = _fmt(value, 3) if unit else _fmt(value, 4)
            rows.append([name, text + (" " + unit if unit != "—" else ""), note])
        if metrics.get("estimator"):
            rows.append(["信噪比估计方法", str(metrics.get("estimator")), "盲估计方法说明"])
        return rows

    def _section_attempts(self) -> None:
        self._h1("四、尝试记录")
        attempts = [a for a in (self.state.get("attempts") or []) if isinstance(a, dict)]
        if not attempts:
            if not self._needs_denoise():
                self._p("数据判定为无需去噪，本次未执行去噪方法尝试。", color=C_GREEN)
            else:
                self._p("本次任务没有去噪尝试记录（流程可能未完成）。", color=C_ORANGE)
            return

        self._p(f"本次共尝试 {len(attempts)} 种去噪方法，以下按执行顺序列出参数、"
                "指标、门禁结果与失败原因。")
        for idx, attempt in enumerate(attempts, start=1):
            method = _as_dict(attempt.get("method"))
            name = method.get("name") or method.get("key") or f"方法 {idx}"
            key = method.get("key", "—")
            self._h2(f"尝试 {idx}：{name}（{key}）")

            rows = [
                ("方法来源", method.get("source", "—")),
                ("目标噪声", "、".join(str(t) for t in (method.get("targets") or [])) or "—"),
                ("强度", _fmt(method.get("strength"), 2)),
                ("运行耗时", f"{_fmt(attempt.get('runtime_s'), 2)} s"),
                ("输出文件", self._rel(attempt.get("output_path"))),
            ]
            self._kv_table(rows)

            params = method.get("params") or {}
            if isinstance(params, dict) and params:
                self._p("方法参数：", bold=True)
                self._table(
                    ["参数", "取值"],
                    [[k, _fmt(v) if isinstance(v, float) else str(v)]
                     for k, v in params.items()],
                    widths=[5.0, 11.4],
                    font_size=8.8,
                )

            metrics = _as_dict(attempt.get("metrics"))
            if metrics:
                self._p("质量指标：", bold=True)
                metric_rows = self._metric_rows(metrics)
                if metric_rows:
                    self._table(["指标", "数值", "参考 / 说明"], metric_rows,
                                widths=[4.8, 2.8, 8.8], font_size=8.8)
                else:
                    self._p("该次尝试未产出可用指标。", color=C_ORANGE)

            gate = _as_dict(attempt.get("gate"))
            if gate:
                self._p("行业门禁：", bold=True)
                summary = ("通过" if gate.get("passed") else "未通过")
                self._p(f"门禁结论：{summary}；得分 {_fmt(gate.get('score'), 1)}；"
                        f"置信度 {_pct(gate.get('confidence'))}",
                        bold=True, color=C_GREEN if gate.get("passed") else C_RED)
                checks = [c for c in (gate.get("checks") or []) if isinstance(c, dict)]
                if checks:
                    self._table(
                        ["检查项", "实际值", "阈值", "结果", "说明"],
                        [[c.get("name", "—"), _fmt(c.get("value"), 3),
                          c.get("threshold", "—"),
                          "通过" if c.get("passed") else "未通过",
                          c.get("note", "")] for c in checks],
                        widths=[3.4, 2.2, 3.0, 1.5, 6.3],
                        font_size=8.5,
                    )
                failures = gate.get("failures") or []
                if failures:
                    self._p("未通过原因：", bold=True, color=C_RED)
                    self._bullets(failures, color=C_RED)
                advice = gate.get("advice") or []
                if advice:
                    self._p("改进建议：", bold=True)
                    self._bullets(advice)

            warnings = attempt.get("warnings") or []
            if warnings:
                self._p("运行警告：", bold=True, color=C_ORANGE)
                self._bullets(warnings, color=C_ORANGE)
            if attempt.get("notes"):
                self._p(f"备注：{attempt.get('notes')}", size=10, color=C_GRAY)

    # ---- 五、质量评价 ----
    def _section_quality(self) -> None:
        self._h1("五、质量评价")
        if not self._needs_denoise():
            self._p("数据判定为无需去噪，本节的质量评价不适用。读取阶段的数据总览"
                    "可作为数据质量良好的佐证（见第一章）。", color=C_GREEN)
            return

        metrics = self._metrics()
        gate = self._gate()
        if not metrics and not gate:
            self._p("本次任务没有可用的质量指标与门禁结果。", color=C_ORANGE)
            return

        if gate and gate.get("passed") is False:
            self._p("本结果未通过行业门禁，以下指标仅作诊断参考。", bold=True, color=C_RED)

        self._h2("指标汇总")
        rows = self._metric_rows(metrics)
        if rows:
            self._table(["指标", "数值", "参考 / 说明"], rows,
                        widths=[4.8, 2.8, 8.8], font_size=9)
        else:
            self._p("没有可用的质量指标。", color=C_ORANGE)

        if gate:
            self._h2("行业门禁结论")
            passed = bool(gate.get("passed"))
            self._p(f"门禁结论：{'通过' if passed else '未通过'}；"
                    f"得分 {_fmt(gate.get('score'), 1)} / 100；"
                    f"置信度 {_pct(gate.get('confidence'))}",
                    bold=True, color=C_GREEN if passed else C_RED)
            checks = [c for c in (gate.get("checks") or []) if isinstance(c, dict)]
            if checks:
                self._table(
                    ["检查项", "实际值", "阈值", "结果", "说明"],
                    [[c.get("name", "—"), _fmt(c.get("value"), 3),
                      c.get("threshold", "—"),
                      "通过" if c.get("passed") else "未通过",
                      c.get("note", "")] for c in checks],
                    widths=[3.4, 2.2, 3.0, 1.5, 6.3],
                    font_size=8.8,
                )
            failures = gate.get("failures") or []
            if failures:
                self._p("未通过门禁的项：", bold=True, color=C_RED)
                self._bullets(failures, color=C_RED)
            advice = gate.get("advice") or []
            if advice:
                self._p("建议：", bold=True)
                self._bullets(advice)

        self._h2("图件对比")
        self._figure("compare", "去噪前后剖面对比（含单道波形）")
        self._figure("diff", "去噪差分（残差）与逐道 RMS 对比")
        self._figure("metrics", "质量指标与门禁检查")
        self._figure("zoom", "局部放大对比")
        if self._find_image("visual") is not None:
            self._figure("visual", "视觉对比与弱信号保留（SSIM / 边缘相关 / 弱区能量）")

    # ---- 六、置信度分析 ----
    def _section_confidence(self) -> None:
        self._h1("六、置信度分析")
        dataset = self.dataset
        source = self.source
        gate = self._gate()
        total = _num(self.state.get("confidence"))
        rows = [
            ("数据读取置信度", _pct(dataset.get("confidence"))),
            ("来源匹配置信度", _pct(source.get("confidence"))),
            ("门禁置信度", _pct(gate.get("confidence"))),
            ("门禁得分", _fmt(gate.get("score"), 1)),
            ("综合置信度", _pct(total)),
        ]
        self._kv_table(rows)
        self._p("综合置信度由读取置信度、来源匹配置信度与门禁置信度（指标可信度 × "
                "门禁余量）共同决定，反映本报告结论的可信程度，不等同于地质解释的"
                "正确性。")
        if total is not None:
            if total >= 0.80:
                level = "高"
            elif total >= 0.60:
                level = "中等"
            elif total >= 0.40:
                level = "偏低"
            else:
                level = "低"
            self._p(f"本次任务综合置信度为 {total:.0%}（{level}），"
                    f"{'可谨慎用于后续处理。' if total >= 0.60 else '建议人工复核后再使用。'}")

        self._h2("局限与注意事项")
        limits: list[str] = []
        if dataset.get("confidence") is not None and (_num(dataset.get("confidence")) or 0) < 0.6:
            limits.append("数据读取置信度偏低，格式/道头/采样可能存在不确定性。")
        if source.get("needs_denoise") and (_num(source.get("confidence")) or 0) < 0.6:
            limits.append("来源匹配置信度偏低，方法规划可能不是最优。")
        if gate and not gate.get("passed"):
            limits.append("去噪结果未通过行业门禁，相关指标不能作为合格依据。")
        if total is not None and total < 0.6:
            limits.append("综合置信度低于 60%，建议人工检查关键剖面与频谱。")
        limits.extend([
            "盲信噪比依赖噪声平稳假设，强非平稳噪声（如涌浪、强面波）下会偏差。",
            "没有干净参考数据时，信号保真度只能间接评估（泄漏、频谱相关）。",
            "大数组在绘图前会按道/时间抽稀，局部极值可能被平滑。",
            "道头缺失或坐标异常时，f-k 速度分析与几何校正的可靠性下降。",
        ])
        self._bullets(limits)

        # 汇总各路警告
        all_warnings: list[str] = []
        for label, items in (
            ("读取", dataset.get("warnings") or []),
            ("来源匹配", source.get("warnings") or []),
            ("任务", self.state.get("warnings") or []),
            ("最佳方法", _get(self._best(), "warnings", default=[]) or []),
        ):
            for item in items:
                text = f"[{label}] {item}"
                if text not in all_warnings:
                    all_warnings.append(text)
        if all_warnings:
            self._h2("警告汇总")
            self._bullets(all_warnings, color=C_ORANGE)

    # ---- 七、结论与建议 ----
    def _section_conclusion(self) -> None:
        self._h1("七、结论与建议")
        if self.verdict_kind == "fail":
            self._p("未通过行业门禁。", size=18, bold=True, color=C_RED,
                    align=WD_ALIGN_PARAGRAPH.CENTER, font=HEAD_FONT)
            self._p(self.verdict_desc, size=11, color=C_RED,
                    align=WD_ALIGN_PARAGRAPH.CENTER)
        else:
            color = C_GREEN if self.verdict_kind in ("ok", "clean") else C_GRAY
            self._p(self.verdict, size=18, bold=True, color=color,
                    align=WD_ALIGN_PARAGRAPH.CENTER, font=HEAD_FONT)
            self._p(self.verdict_desc, size=11, color=color,
                    align=WD_ALIGN_PARAGRAPH.CENTER)

        best = self._best()
        method = _as_dict(best.get("method"))
        metrics = self._metrics()
        gate = self._gate()
        if best:
            parts = [f"主方法：{method.get('name') or method.get('key') or '—'}"]
            if metrics.get("snr_gain_db") is not None:
                parts.append(f"SNR 增益 {_fmt(metrics.get('snr_gain_db'), 2)} dB")
            if metrics.get("snr_out_db") is not None:
                parts.append(f"去噪后 SNR {_fmt(metrics.get('snr_out_db'), 2)} dB")
            if metrics.get("energy_ratio") is not None:
                parts.append(f"能量保持比 {_fmt(metrics.get('energy_ratio'), 3)}")
            if gate:
                parts.append(f"门禁得分 {_fmt(gate.get('score'), 1)}")
            self._p("；".join(parts) + "。")

        # 建议
        advice: list[str] = []
        for item in (gate.get("advice") or []) if gate else []:
            advice.append(str(item))
        for item in (self.source.get("recommended") or []):
            advice.append(f"可优先尝试方法：{item}")
        if self.verdict_kind == "fail":
            advice.append("在通过门禁前，请勿将本结果用于定量解释或成果提交。")
            advice.append("建议补充干净参考/调整参数/更换方法后重试，或降低去噪强度。")
        elif self.verdict_kind == "clean":
            advice.append("保持原始数据，无需去噪；如后续有新数据，可按相同流程复检。")
        elif self.verdict_kind == "ok":
            advice.append("结果已通过门禁，可用于后续处理；仍建议抽查关键剖面。")
        else:
            advice.append("请检查流程日志，确认读取/来源匹配/去噪阶段是否完整执行。")
        # 去重保序
        unique_advice = list(dict.fromkeys(a for a in advice if a))
        if unique_advice:
            self._h2("建议")
            self._bullets(unique_advice)

        history = [h for h in (self.state.get("history") or []) if isinstance(h, dict)]
        if history:
            self._h2("处理历程")
            self._table(
                ["时间", "阶段", "说明"],
                [[h.get("time", "—"), h.get("stage", "—"), h.get("message", "")]
                 for h in history[-12:]],
                widths=[4.2, 2.4, 9.8],
                font_size=8.8,
            )

        self._p("本报告由地震智能去噪插件自动生成，指标口径与门禁阈值见说明.txt。",
                size=9, color=C_GRAY, align=WD_ALIGN_PARAGRAPH.CENTER)

    # ---- 八、文件清单 ----
    def _section_files(self) -> None:
        self._h1("八、文件清单")
        self._p("成果目录中的文件如下。文件清单.csv 与本表一致，可交付时随目录一起提供。")
        manifest = None
        try:
            manifest = pkg.write_manifest(self.job_dir, self.state)
        except Exception as exc:
            self._p(f"（文件清单生成失败：{exc}）", size=9, color=C_ORANGE)

        rows: list[list[Any]] = []
        if manifest is not None and Path(manifest).exists():
            try:
                with Path(manifest).open("r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # 跳过表头
                    for row in reader:
                        if len(row) >= 5:
                            rows.append(row[:5])
            except Exception:
                rows = []
        if not rows:
            # 兜底：直接扫目录
            files = sorted((p.relative_to(self.job_dir), p)
                           for p in self.job_dir.rglob("*") if p.is_file())
            for idx, (rel, path) in enumerate(files, start=1):
                rows.append([idx, path.name, path.suffix.lstrip(".") or "—",
                             pkg.describe_file(rel, self.state), rel.as_posix()])

        if rows:
            self._table(
                ["序号", "文件名", "类型", "说明", "相对路径"],
                rows,
                widths=[1.1, 3.3, 2.2, 6.4, 3.6],
                font_size=8.2,
            )
        else:
            self._p("成果目录中暂无文件。")

    # ======================================================================
    # 组装
    # ======================================================================
    def build(self) -> Path:
        self._ensure_figures()
        # 图表补齐后再落说明.txt，保证第 8 节文件清单和最终 CSV 都收录它；
        # 即使调用方忘记单独调用 write_readme，报告也是自洽的。
        if isinstance(self.state, dict):
            try:
                pkg.write_readme(self.job_dir, self.state)
            except Exception:
                pass
        self._cover()
        self._section_reading()
        self._section_source()
        self._section_plan()
        self._section_attempts()
        self._section_quality()
        self._section_confidence()
        self._section_conclusion()
        self._section_files()

        out = self.job_dir / "05_报告" / pkg.REPORT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(out)
        # 报告生成后再刷一次清单，让 docx 本身出现在清单里
        try:
            pkg.write_manifest(self.job_dir, self.state)
        except Exception:
            pass
        return out


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------
def build_report(job_dir: str | Path, state: Any) -> Path:
    """生成 ``05_报告/去噪过程报告.docx`` 并返回实际路径。

    job_dir 不存在时自动创建；state 缺字段时逐项降级显示，不抛异常。
    """
    builder = _ReportBuilder(job_dir, state)
    return builder.build()


__all__ = ["build_report"]
