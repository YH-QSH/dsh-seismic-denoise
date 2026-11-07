"""成果目录打包：建任务目录、复制原始数据、写文件清单与说明。

契约（docs/接口约定.md §6）：
    create_job_folder(input_path, root=None) -> Path
    copy_original(input_path, job_dir) -> Path
    write_manifest(job_dir, state) -> Path
    write_readme(job_dir, state) -> Path
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils import now_iso, read_json, safe_name

# 成果目录固定五层
JOB_SUBDIRS = ("01_原始数据", "02_读取与诊断", "03_去噪结果", "04_质量评价", "05_报告")
MANIFEST_NAME = "文件清单.csv"
README_NAME = "说明.txt"
REPORT_NAME = "去噪过程报告.docx"

PLUGIN_VERSION = "1.0"


# --------------------------------------------------------------------------
# 目录与原始数据
# --------------------------------------------------------------------------
def create_job_folder(input_path: str | Path, root: str | Path | None = None) -> Path:
    """创建 ``<文件名>_地震去噪_<YYYYmmdd_HHMMSS>`` 成果目录。

    root 为空时放在输入文件同级目录；同名目录已存在时追加序号，避免覆盖。
    """
    src = Path(input_path)
    stem = safe_name(src.stem, maxlen=60) if src.stem else "地震数据"
    base = Path(root) if root is not None else src.parent
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stem}_地震去噪_{stamp}"
    job = base / name
    serial = 1
    while job.exists():
        job = base / f"{name}_{serial:02d}"
        serial += 1
    for sub in JOB_SUBDIRS:
        (job / sub).mkdir(parents=True, exist_ok=True)
    return job


def copy_original(input_path: str | Path, job_dir: str | Path) -> Path:
    """把原始数据复制到 ``01_原始数据/原始数据<后缀>``。

    同盘优先建硬链接（大 SEG-Y 几乎零耗时），失败自动回退到普通复制。
    硬链接与源文件共享数据，删除副本不会影响源文件。
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"原始数据不存在：{src}")
    folder = Path(job_dir) / "01_原始数据"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"原始数据{src.suffix}"
    if dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return dst
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)  # 硬链接：同盘、大文件时最快
    except OSError:
        shutil.copy2(src, dst)  # 跨盘 / 无权限时回退普通复制
    return dst


# --------------------------------------------------------------------------
# 文件类型与说明
# --------------------------------------------------------------------------
_EXT_TYPES = {
    ".segy": "SEG-Y 地震数据",
    ".sgy": "SEG-Y 地震数据",
    ".npy": "NumPy 数组",
    ".npz": "NumPy 压缩包",
    ".mat": "MATLAB 数据",
    ".h5": "HDF5 数据",
    ".hdf5": "HDF5 数据",
    ".raw": "裸二进制数据",
    ".dat": "二进制数据",
    ".bin": "二进制数据",
    ".png": "图片",
    ".jpg": "图片",
    ".jpeg": "图片",
    ".docx": "Word 报告",
    ".csv": "CSV 表格",
    ".json": "JSON 数据",
    ".txt": "文本",
    ".log": "日志",
    ".pt": "模型权重",
    ".pth": "模型权重",
}

_NAME_DESCRIPTIONS = {
    "数据总览.png": "数据总览：剖面、振幅分布、单道 RMS 能量",
    "频率谱.png": "平均振幅谱（含 5–80 Hz 信号带）",
    "振幅谱.png": "平均振幅谱（含 5–80 Hz 信号带）",
    "fk谱.png": "F-K 二维振幅谱",
    "F-K谱.png": "F-K 二维振幅谱",
    "道头表.csv": "道头字段统计表（一字段一行）",
    "道头分析.png": "关键道头随道号变化曲线",
    "数据集信息.json": "数据读取与诊断信息（格式、采样、振幅、置信度、警告）",
    "数据缓存.npz": "中间缓存（数据矩阵 + dt/dz），可删除，不影响成果",
    "来源匹配.json": "来源匹配与数据清洁度评估结果",
    "来源匹配图.png": "来源匹配与噪声类型概率图",
    "去噪后剖面.png": "去噪前后剖面对比图",
    "去噪差分.png": "去噪差分（残差）与逐道 RMS 对比图",
    "质量评价.json": "质量指标、门禁检查与置信度",
    "指标对比.png": "质量指标与门禁检查图",
    "局部放大对比.png": "局部时间/道窗的放大对比图",
    "去噪过程报告.docx": "完整 Word 报告（本文件）",
    "文件清单.csv": "本清单：所有成果文件及用途",
    "说明.txt": "成果目录说明与复现方法",
    "任务状态.json": "任务状态（中间文件）",
    "运行日志.txt": "运行日志（中间文件）",
}


def _file_type(path: Path) -> str:
    """按后缀给文件分类（写清单用）。"""
    return _EXT_TYPES.get(path.suffix.lower(), "其他")


def _norm_label(text: str) -> str:
    """归一化方法名，便于用文件名里的短标签匹配方法全名。"""
    return "".join(ch for ch in str(text).lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _method_label(state: dict | None, token: str) -> str:
    """把 ``去噪结果_<token>.segy`` 里的 token 还原成方法显示名。

    先在各次尝试、最终结果和计划里找 key/name 与 token 匹配的方法，
    匹配不到就直接用 token，绝不套用其它方法的名称。
    """
    norm = _norm_label(token)
    candidates: list[dict] = []
    if isinstance(state, dict):
        for attempt in state.get("attempts") or []:
            method = attempt.get("method") if isinstance(attempt, dict) else None
            if isinstance(method, dict):
                candidates.append(method)
        best = state.get("best")
        if isinstance(best, dict) and isinstance(best.get("method"), dict):
            candidates.append(best["method"])
        for spec in state.get("plan") or []:
            if isinstance(spec, dict):
                candidates.append(spec)
    for candidate in candidates:
        for value in (candidate.get("key"), candidate.get("name"), candidate.get("method")):
            if not value:
                continue
            norm_value = _norm_label(value)
            if norm_value and (norm in norm_value or norm_value in norm):
                return str(candidate.get("name") or candidate.get("key") or token)
    return token or "未知方法"


def _describe(rel: Path, state: dict | None = None) -> str:
    """给单个成果文件写一句中文说明。"""
    path = Path(rel)
    name = path.name
    folder = path.parts[0] if len(path.parts) > 1 else ""
    if name.startswith("原始数据"):
        return "原始输入数据（只读副本，供追溯）"
    if name.startswith("去噪结果_") and path.suffix.lower() in (".segy", ".sgy", ".npy"):
        token = name[len("去噪结果_"):]
        if path.suffix:
            token = token[: -len(path.suffix)]
        method = _method_label(state, token)
        return f"去噪后的地震数据（方法：{method}）" if method else "去噪后的地震数据"
    if name in _NAME_DESCRIPTIONS:
        return _NAME_DESCRIPTIONS[name]
    # 按目录给兜底说明
    folder_defaults = {
        "01_原始数据": "原始数据副本",
        "02_读取与诊断": "数据读取与诊断成果",
        "03_去噪结果": "去噪结果数据/图件",
        "04_质量评价": "质量评价成果",
        "05_报告": "报告与清单",
    }
    if folder in folder_defaults:
        return folder_defaults[folder]
    return f"成果文件（{_file_type(path)}）"


# --------------------------------------------------------------------------
# 文件清单
# --------------------------------------------------------------------------
def write_manifest(job_dir: str | Path, state: dict | None = None) -> Path:
    """扫描成果目录，写 ``05_报告/文件清单.csv``。

    列：序号,文件名,类型,说明,相对路径。清单本身不列入清单。
    用 utf-8-sig 编码，Excel 直接打开不乱码。
    """
    job = Path(job_dir)
    report_dir = job / "05_报告"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / MANIFEST_NAME

    files = []
    for path in job.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(job)
        if rel.as_posix() == f"05_报告/{MANIFEST_NAME}":
            continue
        if path.name.startswith("~$") or path.name.startswith("."):
            continue
        files.append((rel, path))
    files.sort(key=lambda item: item[0].as_posix())

    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["序号", "文件名", "类型", "说明", "相对路径"])
        for idx, (rel, path) in enumerate(files, start=1):
            writer.writerow([
                idx,
                path.name,
                _file_type(path),
                _describe(rel, state),
                rel.as_posix(),
            ])
    return out


# --------------------------------------------------------------------------
# 中文说明
# --------------------------------------------------------------------------
def _load_state(state: Any) -> dict:
    """state 可以是 dict，也可以是 JSON 路径。"""
    if isinstance(state, dict):
        return state
    if isinstance(state, (str, Path)):
        data = read_json(state, default=None)
        return data if isinstance(data, dict) else {}
    return {}


def _verdict_text(state: dict) -> str:
    """从 state 中提取一句人话结论。"""
    source = state.get("source") or {}
    needs = source.get("needs_denoise", None) if isinstance(source, dict) else None
    gate = state.get("gate") or {}
    attempts = state.get("attempts") or []
    if needs is False:
        return "无需去噪（数据质量良好）"
    if isinstance(gate, dict) and gate.get("passed"):
        return "达标（已通过行业门禁）"
    if attempts or gate:
        return "未达标（未通过行业门禁）"
    return "未完成（尚无质量门禁结果）"


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_readme(job_dir: str | Path, state: dict | None = None) -> Path:
    """写 ``05_报告/说明.txt``：目录用途、逐文件说明、复现命令、门禁口径。"""
    job = Path(job_dir)
    state_data = _load_state(state)
    report_dir = job / "05_报告"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / README_NAME

    dataset = state_data.get("dataset") or {}
    source = state_data.get("source") or {}
    input_path = state_data.get("input") or ""
    created = state_data.get("created_at") or now_iso()
    version = state_data.get("version") or PLUGIN_VERSION
    stage = state_data.get("stage") or "done"

    lines: list[str] = []
    lines.append("地震智能去噪成果说明")
    lines.append("=" * 46)
    lines.append(f"生成时间：{created}")
    lines.append(f"原始数据：{input_path or '（未记录）'}")
    lines.append(f"处理阶段：{stage}")
    lines.append(f"插件版本：{version}")
    lines.append(f"判定结论：{_verdict_text(state_data)}")
    if isinstance(dataset, dict) and dataset:
        lines.append(
            f"数据规模：{dataset.get('n_samples', '—')} 采样点 × "
            f"{dataset.get('n_traces', '—')} 道，"
            f"dt = {dataset.get('dt_ms', '—')} ms，dz = {dataset.get('dz_m', '—')} m"
        )
    if isinstance(source, dict) and source:
        lines.append(
            f"来源匹配：{source.get('domain', '未知')} / "
            f"{source.get('acquisition', '未知')}，"
            f"置信度 {float(source.get('confidence') or 0):.0%}"
        )
    lines.append("")

    lines.append("一、成果目录")
    lines.append("-" * 46)
    folder_desc = {
        "01_原始数据": "原始输入数据的只读副本，供追溯",
        "02_读取与诊断": "格式识别、道头分析、频谱/F-K 谱、来源匹配",
        "03_去噪结果": "去噪后的地震数据、剖面对比与差分图",
        "04_质量评价": "质量指标、行业门禁结果、局部对比图",
        "05_报告": "Word 报告、文件清单与本说明",
    }
    for folder in JOB_SUBDIRS:
        lines.append(f"{folder}/    {folder_desc[folder]}")
    lines.append("")

    lines.append("二、文件用途")
    lines.append("-" * 46)
    for rel, path in sorted(
        ((p.relative_to(job), p) for p in job.rglob("*") if p.is_file()),
        key=lambda item: item[0].as_posix(),
    ):
        if rel.as_posix() == f"05_报告/{README_NAME}":
            continue
        lines.append(f"{rel.as_posix()}    —— {_describe(rel, state_data)}")
    lines.append("")

    lines.append("三、如何复现")
    lines.append("-" * 46)
    lines.append("在插件根目录执行（Windows 命令提示符）：")
    lines.append(f'    cd /d "{_plugin_root()}"')
    sample_input = input_path or str(job / "01_原始数据")
    lines.append(
        f'    python -m engine.cli --input "{sample_input}" '
        f'--workdir "{job}"'
    )
    lines.append("")
    lines.append("如果当前 CLI 版本支持分阶段执行，可依次运行 read / assess /")
    lines.append("denoise / report 子命令（具体参数以 --help 输出为准）：")
    lines.append('    python -m engine.cli --help')
    lines.append("")
    lines.append("一键重跑本任务时，建议输出到新目录，避免覆盖已有成果。")
    lines.append("")

    lines.append("四、质量门禁参考")
    lines.append("-" * 46)
    lines.append("- SNR 增益 ≥ 3.0 dB（硬指标）")
    lines.append("- 能量保持比在 0.30 ~ 1.05")
    lines.append("- 信号泄漏 ≤ 0.40")
    lines.append("- 5–80 Hz 信号带频谱相关 ≥ 0.80")
    lines.append("- 削顶比例 ≤ 1e-3")
    lines.append("- 去噪后盲信噪比 ≥ 10.0 dB（盲估计不可靠时仅作软检查）")
    lines.append("未通过门禁的结果不建议直接用于定量解释；")
    lines.append("干净数据（needs_denoise=False）不进入去噪门禁。")
    lines.append("")

    lines.append("五、注意事项")
    lines.append("-" * 46)
    lines.append("- 本目录由程序自动生成；修改 01_原始数据 中的副本不会影响源文件。")
    lines.append("- 若原始数据与任务目录在同一磁盘，01_原始数据 可能是硬链接，")
    lines.append("  占用的额外空间接近 0；删除副本不会删除源文件。")
    lines.append("- 数据缓存.npz、任务状态.json、运行日志.txt 是中间文件，")
    lines.append("  可在交付时删除，不影响报告与图件。")
    lines.append("- 盲信噪比依赖平稳噪声假设，强非平稳噪声下置信度会下降。")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# 供 docx_report 等模块复用的公开别名
describe_file = _describe

__all__ = [
    "JOB_SUBDIRS",
    "MANIFEST_NAME",
    "README_NAME",
    "REPORT_NAME",
    "create_job_folder",
    "copy_original",
    "write_manifest",
    "write_readme",
    "describe_file",
]
