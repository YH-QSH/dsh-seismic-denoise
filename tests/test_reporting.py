"""reporting 层自测：合成地震数据 → 完整成果目录 → Word 报告。

运行方式（pytest 未安装时直接跑脚本）：
    python tests/test_reporting.py

覆盖：
1. 8 个契约图表函数 + 抽稀/尺寸/中文字体回退；
2. create_job_folder / copy_original / write_manifest / write_readme；
3. 达标报告的 9 节结构、图片、表格、字体、docx 可重新打开；
4. 未达标报告必须出现“未通过行业门禁”；
5. 干净数据（needs_denoise=False）的“无需去噪”报告。

所有产物只写到 tests/_out/，不污染插件根目录。
"""
from __future__ import annotations

import csv
import math
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402

from engine.model import (  # noqa: E402
    GateCheck,
    GateResult,
    HeaderTable,
    MethodResult,
    MethodSpec,
    QualityMetrics,
    SeismicDataset,
    SourceAssessment,
    dumps,
)
from engine.reporting import figures as F  # noqa: E402
from engine.reporting import package as pkg  # noqa: E402
from engine.reporting.docx_report import build_report  # noqa: E402
from engine.utils import now_iso  # noqa: E402

OUT = ROOT / "tests" / "_out"
NS, NTR = 4000, 436
DT, DZ = 0.001, 12.5


# --------------------------------------------------------------------------
# 合成数据
# --------------------------------------------------------------------------
def _ricker(nt: int, f0: float, dt: float) -> np.ndarray:
    """雷克子波（奇数长度，中心在 nt//2）。"""
    t = (np.arange(nt) - nt // 2) * dt
    arg = (math.pi * f0 * t) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def _add_event(
    volume: np.ndarray,
    wavelet: np.ndarray,
    centers: np.ndarray,
    amplitude: float,
) -> None:
    """按每道中心样点把子波叠到剖面上。"""
    half = wavelet.size // 2
    for trace, center in enumerate(centers):
        lo = int(center) - half
        hi = lo + wavelet.size
        if lo < 0:
            wavelet_slice = wavelet[-lo:]
            lo = 0
        else:
            wavelet_slice = wavelet
        if hi > volume.shape[0]:
            hi = volume.shape[0]
            wavelet_slice = wavelet_slice[: hi - lo]
        if hi > lo:
            volume[lo:hi, trace] += amplitude * wavelet_slice


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (clean, noisy, denoised)，形状 (4000, 436) float32。"""
    rng = np.random.default_rng(20261106)
    offsets = np.arange(NTR, dtype=np.float64) * DZ

    clean = np.zeros((NS, NTR), dtype=np.float64)
    wave = _ricker(161, 26.0, DT)
    events = (
        (0.28, 0.42, 1800.0),
        (0.52, 0.40, 2000.0),
        (0.78, 0.38, 2200.0),
        (1.06, 0.34, 2400.0),
        (1.36, 0.30, 2600.0),
    )
    for t0, amplitude, velocity in events:
        times = np.sqrt(t0 ** 2 + (offsets / velocity) ** 2)
        _add_event(clean, wave, np.rint(times / DT), amplitude)

    # 面波：低速线性同相轴（视速度约 600 m/s，避免空间假频），只进入含噪数据
    ground = np.zeros((NS, NTR), dtype=np.float64)
    long_wave = _ricker(321, 8.0, DT)
    _add_event(ground, long_wave, np.rint((0.18 + offsets / 600.0) / DT), 0.55)

    t = np.arange(NS, dtype=np.float64) * DT
    powerline = 0.03 * np.sin(2.0 * math.pi * 50.0 * t)[:, None]
    random_noise = rng.normal(0.0, 0.03, size=(NS, NTR))

    noisy = clean + ground + powerline + random_noise
    scale = max(1.0, float(np.max(np.abs(noisy))))
    noisy = (noisy / scale).astype(np.float32)
    clean = (clean / scale).astype(np.float32)

    # 构造一个“看起来合格”的去噪结果：有效信号完整保留，各类噪声压到 25% 振幅。
    # 这里不调用真实去噪算法——reporting 层测试只需要可信的指标与图像。
    denoised = (clean + 0.25 * (ground + powerline + random_noise)) / scale
    denoised = denoised.astype(np.float32)
    return clean, noisy, denoised


def _make_dataset(data: np.ndarray, path: Path) -> SeismicDataset:
    """构造带 3 个道头字段的 SeismicDataset。"""
    headers = HeaderTable()
    trace = np.arange(1, NTR + 1)
    headers.add("trace_seq", "道序号", 1, 4, "int32", trace, "从 1 开始的顺序号")
    headers.add("cdp", "CDP 号", 21, 4, "int32", np.repeat(np.arange(1, 110), 4)[:NTR],
                "合成数据 CDP")
    headers.add("offset", "偏移距", 37, 4, "float32",
                np.arange(NTR) * DZ, "单位 m")
    return SeismicDataset(
        path=str(path),
        format="SEGY",
        data=np.asarray(data, dtype=np.float32),
        dt=DT,
        dz=DZ,
        headers=headers,
        text_header=(
            "*** 合成地震数据（仅用于 reporting 层自测） ***\n"
            "4000 samples, 436 traces, dt=1ms, dz=12.5m\n"
            "C01 CLIENT TEST DATA - NOT FOR INTERPRETATION"
        ),
        meta={"synthetic": True, "note": "reporting 自测"},
        confidence=0.93,
        warnings=["合成数据：仅用于验证报告与图表渲染。"],
        source_kind="seg-y",
    )


def _metrics_info(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> dict:
    """用真值计算一组可信的质量指标（仅测试用，不依赖 engine.quality）。"""
    clean64 = np.asarray(clean, dtype=np.float64)
    noise_in = np.asarray(noisy, dtype=np.float64) - clean64
    noise_out = np.asarray(denoised, dtype=np.float64) - clean64

    def _snr(noise: np.ndarray) -> float:
        signal_power = float(np.sum(clean64 ** 2))
        noise_power = float(np.sum(noise ** 2))
        return 10.0 * math.log10(signal_power / max(noise_power, 1e-12))

    snr_in = _snr(noise_in)
    snr_out = _snr(noise_out)
    energy_ratio = float(np.sum(np.asarray(denoised, dtype=np.float64) ** 2)
                         / max(float(np.sum(np.asarray(noisy, dtype=np.float64) ** 2)), 1e-12))
    signal_leakage = float(np.sum(noise_out ** 2) / max(float(np.sum(clean64 ** 2)), 1e-12))

    freqs = np.fft.rfftfreq(NS, d=DT)
    band = (freqs >= 5.0) & (freqs <= 80.0)
    spec_clean = np.abs(np.fft.rfft(clean64, axis=0)).mean(axis=1)[band]
    spec_den = np.abs(np.fft.rfft(
        np.asarray(denoised, dtype=np.float64), axis=0)).mean(axis=1)[band]
    spectrum_corr = float(np.corrcoef(spec_clean, spec_den)[0, 1])
    coherence_residual = float(
        np.sum(noise_out * clean64) / math.sqrt(
            max(float(np.sum(noise_out ** 2) * np.sum(clean64 ** 2)), 1e-12))
    )

    gains = []
    for chunk in np.array_split(np.arange(NS), 6):
        c = clean64[chunk]
        ni = noise_in[chunk]
        no = noise_out[chunk]
        sig = float(np.sum(c ** 2))
        gains.append(10.0 * math.log10(max(sig, 1e-12) / max(float(np.sum(no ** 2)), 1e-12))
                     - 10.0 * math.log10(max(sig, 1e-12) / max(float(np.sum(ni ** 2)), 1e-12)))
    metrics = QualityMetrics(
        snr_in_db=round(snr_in, 3),
        snr_out_db=round(snr_out, 3),
        snr_gain_db=round(snr_out - snr_in, 3),
        reference_snr_in_db=round(snr_in + 0.4, 3),
        reference_snr_out_db=round(snr_out + 0.2, 3),
        reference_gain_db=round(snr_out - snr_in - 0.2, 3),
        noise_before_db=round(10.0 * math.log10(max(float(np.mean(noise_in ** 2)), 1e-12)), 3),
        noise_after_db=round(10.0 * math.log10(max(float(np.mean(noise_out ** 2)), 1e-12)), 3),
        noise_reduction_db=round(
            10.0 * math.log10(max(float(np.mean(noise_in ** 2)), 1e-12))
            - 10.0 * math.log10(max(float(np.mean(noise_out ** 2)), 1e-12)), 3),
        energy_ratio=round(energy_ratio, 4),
        signal_leakage=round(signal_leakage, 4),
        spectrum_corr=round(spectrum_corr, 4),
        coherence_residual=round(coherence_residual, 4),
        clip_ratio=float(np.mean(np.abs(np.asarray(denoised)) >= 0.9999)),
        local_snr_gain_db=round(float(np.mean(gains)), 3),
        estimator="盲估计（道集统计）",
        extra={"reference": "合成数据真值", "window": "6 段"},
    )
    return metrics.to_info()


def _failed_metrics_info() -> dict:
    """一个未达标的指标组（供尝试记录展示红色失败原因）。"""
    return QualityMetrics(
        snr_in_db=4.2,
        snr_out_db=5.4,
        snr_gain_db=1.2,
        energy_ratio=0.24,
        signal_leakage=0.46,
        spectrum_corr=0.74,
        coherence_residual=0.31,
        clip_ratio=0.0,
        local_snr_gain_db=0.9,
        estimator="盲估计（道集统计）",
    ).to_info()


def _gate_info(metrics: dict, passed: bool = True) -> dict:
    """按契约门禁阈值构造 GateResult。"""
    checks = [
        GateCheck("SNR 增益", metrics.get("snr_gain_db"), "≥ 3.0 dB",
                  float(metrics.get("snr_gain_db") or 0) >= 3.0,
                  "硬指标"),
        GateCheck("能量保持比", metrics.get("energy_ratio"), "0.30 ~ 1.05",
                  0.30 <= float(metrics.get("energy_ratio") or 0) <= 1.05,
                  "过低说明有效能量被削弱"),
        GateCheck("信号泄漏", metrics.get("signal_leakage"), "≤ 0.40",
                  float(metrics.get("signal_leakage") or 0) <= 0.40,
                  "残差中有效信号占比"),
        GateCheck("频谱相关", metrics.get("spectrum_corr"), "≥ 0.80",
                  float(metrics.get("spectrum_corr") or 0) >= 0.80,
                  "5–80 Hz 信号带"),
        GateCheck("削顶比例", metrics.get("clip_ratio"), "≤ 1e-3",
                  float(metrics.get("clip_ratio") or 0) <= 1e-3,
                  "保护振幅"),
        GateCheck("去噪后 SNR", metrics.get("snr_out_db"), "≥ 10.0 dB（软）",
                  float(metrics.get("snr_out_db") or 0) >= 10.0,
                  "盲估计不可靠时为软检查"),
    ]
    failures = [f"{c.name} {c.value:g} 未满足 {c.threshold}"
                for c in checks if not c.passed]
    gate = GateResult(
        passed=passed and not failures,
        score=92.0 if passed and not failures else 55.0,
        checks=checks,
        failures=failures,
        confidence=0.85 if passed else 0.42,
        advice=[] if passed else [
            "降低去噪强度，避免过度压制有效信号。",
            "检查面波视速度范围，扩大保护带。",
        ],
    )
    return gate.to_info()


# --------------------------------------------------------------------------
# 小工具：PNG 尺寸 / docx 全文本
# --------------------------------------------------------------------------
def _png_size(path: Path) -> tuple[int, int]:
    with Path(path).open("rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} 不是 PNG"
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


def _docx_text(doc) -> str:
    chunks = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    for section in doc.sections:
        chunks.extend(p.text for p in section.footer.paragraphs)
        chunks.extend(p.text for p in section.header.paragraphs)
    return "\n".join(chunks)


# --------------------------------------------------------------------------
# 1. 图表
# --------------------------------------------------------------------------
def test_figures() -> None:
    clean, noisy, denoised = _synthetic()
    fig_dir = OUT / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    F.setup_style()
    import matplotlib.pyplot as plt

    assert plt.rcParams["savefig.dpi"] == F.DPI == 120
    thinned, _, _ = F._thin(noisy)
    assert max(thinned.shape) <= F.MAX_DIM, f"抽稀失败：{thinned.shape}"

    ds = _make_dataset(noisy, OUT / "输入" / "合成地震记录.segy")
    assessment = _assessment().to_info()
    metrics = _metrics_info(clean, noisy, denoised)
    gate = _gate_info(metrics, passed=True)

    outputs = {
        "overview": F.fig_data_overview(noisy, DT, DZ, fig_dir / "数据总览.png"),
        "header": F.fig_header_analysis(ds, fig_dir / "道头分析.png"),
        "spectrum": F.fig_spectrum(noisy, DT, fig_dir / "频率谱.png"),
        "fk": F.fig_fk(noisy, DT, DZ, fig_dir / "fk谱.png"),
        "source": F.fig_source_assessment(ds, assessment, fig_dir / "来源匹配图.png"),
        "compare": F.fig_compare(noisy, denoised, DT, DZ, fig_dir / "去噪后剖面.png"),
        "multimetrics": F.fig_metrics({"metrics": metrics, "gate": gate},
                                      fig_dir / "指标对比.png"),
        "zoom": F.fig_local_zoom(noisy, denoised, DT, DZ, fig_dir / "局部放大对比.png"),
        "residual": F.fig_residual(noisy, denoised, DT, DZ, fig_dir / "去噪差分.png"),
    }
    for name, path in outputs.items():
        assert isinstance(path, Path), f"{name} 未返回 Path"
        assert path.exists() and path.stat().st_size > 5000, f"{name} 输出异常：{path}"
        size = _png_size(path)
        assert size == (1600, 1000), f"{name} 尺寸 {size} != (1600, 1000)"
    print(f"[figures] 9 张图全部生成，1600×1000，中文字体={F.cjk_available()}")


def _assessment() -> SourceAssessment:
    """合成来源匹配结果（陆地二维反射数据 + 面波/随机噪声）。"""
    return SourceAssessment(
        domain="陆地反射地震",
        acquisition="可控震源 + 检波器排列",
        geometry_type="二维",
        confidence=0.86,
        evidence=[
            "道头包含 CDP/偏移距，排列规则，判定为二维反射采集。",
            "低频低速线性同相轴明显，符合面波特征。",
            "5–80 Hz 主频与反射波一致。",
        ],
        geometry={"n_traces": NTR, "dz_m": DZ, "spread_m": DZ * NTR},
        cleanliness={"dead_traces": 0, "spikes": 0.0, "dc_offset": 0.001},
        noise_types={"面波": 0.82, "随机噪声": 0.55, "工频干扰": 0.30},
        needs_denoise=True,
        recommended=["F-K 滤波", "带通滤波", "f-x 反褶积"],
        clean_reasons=[],
        warnings=["合成数据：来源匹配为测试桩。"],
        notes="二维单边排列，覆盖次数约 4 次。",
    )


# --------------------------------------------------------------------------
# 2. 打包 + 达标报告
# --------------------------------------------------------------------------
def _build_main_job() -> tuple[Path, dict, Path]:
    """生成完整成果目录与达标报告，返回 (job_dir, state, report_path)。"""
    clean, noisy, denoised = _synthetic()
    jobs_root = OUT / "jobs"
    if jobs_root.exists():
        shutil.rmtree(jobs_root)
    jobs_root.mkdir(parents=True)

    src_dir = OUT / "输入"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "合成地震记录.segy"
    if not src.exists():
        # 1 MB 假 SEG-Y，验证硬链接/复制与清单
        src.write_bytes(b"\x40" * (1024 * 1024) + b"SYNTHETIC")

    job = pkg.create_job_folder(src, root=jobs_root)
    assert re.fullmatch(r"合成地震记录_地震去噪_\d{8}_\d{6}(_\d{2})?", job.name), job.name
    for sub in pkg.JOB_SUBDIRS:
        assert (job / sub).is_dir(), f"缺少目录 {sub}"

    original = pkg.copy_original(src, job)
    assert original.name == "原始数据.segy"
    assert original.read_bytes()[:4] == b"\x40\x40\x40\x40"

    ds = _make_dataset(noisy, src)
    ds.save_cache(job / "02_读取与诊断")
    source = _assessment().to_info()
    (job / "02_读取与诊断" / "来源匹配.json").write_text(
        dumps(source), encoding="utf-8")

    metrics = _metrics_info(clean, noisy, denoised)
    gate = _gate_info(metrics, passed=True)

    fk_spec = MethodSpec(
        key="fk_filter",
        name="F-K 滤波",
        params={"keep_velocity": 800.0, "freq_min": 2.0, "taper": 0.10},
        rationale="面波视速度低于 800 m/s，用扇形滤波压制且保护反射同相轴。",
        targets=["面波", "线性干扰"],
        strength=0.7,
        priority=1,
        source="rule",
    )
    bp_spec = MethodSpec(
        key="bandpass",
        name="带通滤波",
        params={"low_hz": 8.0, "high_hz": 70.0, "poles": 4},
        rationale="压制低频面波与高频随机噪声，作为保守备选。",
        targets=["低频", "高频"],
        strength=0.5,
        priority=2,
        source="rule",
    )

    fail_metrics = _failed_metrics_info()
    fail_gate = _gate_info(fail_metrics, passed=False)
    bp_result = MethodResult(
        spec=bp_spec,
        metrics=QualityMetrics(**{k: v for k, v in fail_metrics.items()
                                  if k in QualityMetrics.__dataclass_fields__}),
        gate=GateResult(
            passed=False,
            score=55.0,
            checks=[GateCheck(**{k: v for k, v in c.items()
                                 if k in GateCheck.__dataclass_fields__})
                    for c in fail_gate["checks"]],
            failures=fail_gate["failures"],
            confidence=fail_gate["confidence"],
            advice=fail_gate["advice"],
        ),
        runtime_s=1.83,
        warnings=["带通滤波对低速面波压制有限。"],
        notes="第一轮保守尝试，未通过门禁。",
        output_path=str(job / "03_去噪结果" / "去噪结果_带通滤波.segy"),
    )
    fk_result = MethodResult(
        spec=fk_spec,
        metrics=QualityMetrics(**{k: v for k, v in metrics.items()
                                  if k in QualityMetrics.__dataclass_fields__}),
        gate=GateResult(
            passed=True,
            score=gate["score"],
            checks=[GateCheck(**{k: v for k, v in c.items()
                                 if k in GateCheck.__dataclass_fields__})
                    for c in gate["checks"]],
            failures=[],
            confidence=gate["confidence"],
            advice=["结果可用于后续处理，建议抽查关键剖面。"],
        ),
        runtime_s=2.41,
        warnings=[],
        notes="扇形滤波参数经一次回退后稳定。",
        output_path=str(job / "03_去噪结果" / "去噪结果_fk滤波.segy"),
    )

    # 去噪结果数据（假 SEG-Y 头 + 截断内容，仅用于交付演示）
    out_segy = job / "03_去噪结果" / "去噪结果_fk滤波.segy"
    out_segy.write_bytes(b"\x40" * 4096)
    (job / "03_去噪结果" / "去噪结果_带通滤波.segy").write_bytes(b"\x40" * 2048)

    # 图件：按契约目录直接生成（顺带验证在成果目录中的真实路径）
    F.fig_data_overview(noisy, DT, DZ, job / "02_读取与诊断" / "数据总览.png")
    F.fig_header_analysis(ds, job / "02_读取与诊断" / "道头分析.png")
    F.fig_spectrum(noisy, DT, job / "02_读取与诊断" / "频率谱.png")
    F.fig_fk(noisy, DT, DZ, job / "02_读取与诊断" / "fk谱.png")
    F.fig_source_assessment(ds, source, job / "02_读取与诊断" / "来源匹配图.png")
    F.fig_compare(noisy, denoised, DT, DZ, job / "03_去噪结果" / "去噪后剖面.png")
    F.fig_residual(noisy, denoised, DT, DZ, job / "03_去噪结果" / "去噪差分.png")
    F.fig_metrics({"metrics": metrics, "gate": gate},
                  job / "04_质量评价" / "指标对比.png")
    F.fig_local_zoom(noisy, denoised, DT, DZ,
                     job / "04_质量评价" / "局部放大对比.png")

    a1 = bp_result.to_info()
    a2 = fk_result.to_info()
    state = {
        "version": "1.0",
        "job_id": job.name,
        "input": str(src),
        "workdir": str(job),
        "stage": "done",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "dataset": ds.to_info(),
        "source": source,
        "plan": [fk_spec.to_info(), bp_spec.to_info()],
        "attempts": [a1, a2],
        "best": a2,
        "gate": gate,
        "confidence": 0.87,
        "outputs": {
            "denoised": str(out_segy),
            "report": str(job / "05_报告" / pkg.REPORT_NAME),
            "folder": str(job),
        },
        "history": [
            {"time": now_iso(), "stage": "read", "message": "读取 SEG-Y 完成，436 道。"},
            {"time": now_iso(), "stage": "assess", "message": "来源匹配：陆地二维反射，建议 F-K。"},
            {"time": now_iso(), "stage": "denoise", "message": "带通滤波未通过门禁，回退 F-K。"},
            {"time": now_iso(), "stage": "done", "message": "F-K 滤波通过门禁，任务完成。"},
        ],
        "warnings": ["合成数据：仅用于 reporting 层自测。"],
    }
    (job / "任务状态.json").write_text(dumps(state), encoding="utf-8")
    (job / "04_质量评价" / "质量评价.json").write_text(
        dumps({"metrics": metrics, "gate": gate, "confidence": 0.87}),
        encoding="utf-8")

    report = build_report(job, state)
    pkg.write_readme(job, state)
    pkg.write_manifest(job, state)
    return job, state, report


def test_package_and_report() -> None:
    job, state, report = _build_main_job()
    assert report == job / "05_报告" / pkg.REPORT_NAME
    assert report.exists() and report.stat().st_size > 50 * 1024, (
        f"docx 太小：{report.stat().st_size} 字节")

    # 清单：列名、内容、docx/说明均在册
    manifest = job / "05_报告" / pkg.MANIFEST_NAME
    assert manifest.exists()
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["序号", "文件名", "类型", "说明", "相对路径"], rows[0]
    assert len(rows) >= 12, f"清单行数过少：{len(rows) - 1}"
    listed = {row[1] for row in rows[1:]}
    assert pkg.REPORT_NAME in listed, listed
    assert pkg.README_NAME in listed, listed
    assert "原始数据.segy" in listed, listed

    # 每个结果文件的说明应来自它自己的方法名，不能都套用最终方法
    descriptions = {row[1]: row[3] for row in rows[1:]}
    assert "F-K" in descriptions["去噪结果_fk滤波.segy"], descriptions
    assert "带通滤波" in descriptions["去噪结果_带通滤波.segy"], descriptions

    readme = (job / "05_报告" / pkg.README_NAME).read_text(encoding="utf-8")
    assert "如何复现" in readme and "python -m engine.cli" in readme
    assert "质量门禁参考" in readme and "SNR 增益" in readme

    # docx 重新打开与结构检查
    doc = Document(str(report))
    text = _docx_text(doc)
    for section in ("数据读取", "来源匹配", "去噪决策", "尝试记录",
                    "质量评价", "置信度分析", "结论与建议", "文件清单"):
        assert section in text, f"报告缺少章节：{section}"
    assert "达标" in text
    assert "未通过行业门禁" not in text, "达标报告不应出现未通过门禁字样"
    assert len(doc.paragraphs) >= 80, f"段落过少：{len(doc.paragraphs)}"
    assert len(doc.tables) >= 10, f"表格过少：{len(doc.tables)}"
    assert len(doc.inline_shapes) >= 8, f"图片过少：{len(doc.inline_shapes)}"
    assert "宋体" in {run.font.name for p in doc.paragraphs for run in p.runs
                      if run.text.strip()}
    # 图片宽度 16cm（EMU：1cm = 360000）
    for shape in doc.inline_shapes:
        assert abs(shape.width.cm - 16.0) < 0.05, shape.width.cm
    print(f"[docx] 达标报告 {report.stat().st_size / 1024:.0f} KB，"
          f"{len(doc.paragraphs)} 段，{len(doc.tables)} 表，{len(doc.inline_shapes)} 图")

    # 删除指标图后重建，验证 _ensure_figures 的自动补图
    metrics_png = job / "04_质量评价" / "指标对比.png"
    metrics_png.unlink()
    report2 = build_report(job, state)
    assert report2.exists()
    assert metrics_png.exists(), "重建报告未自动补出 指标对比.png"
    print("[docx] 自动补图验证通过")


def test_fail_report() -> None:
    """未达标：必须醒目出现“未通过行业门禁”，且不能崩溃。"""
    clean, noisy, denoised = _synthetic()
    _ = clean, denoised
    jobs_root = OUT / "jobs_fail"
    if jobs_root.exists():
        shutil.rmtree(jobs_root)
    src = OUT / "输入" / "合成地震记录.segy"
    job = pkg.create_job_folder(src, root=jobs_root)
    ds = _make_dataset(noisy, src)
    ds.save_cache(job / "02_读取与诊断")

    fail_metrics = _failed_metrics_info()
    fail_gate = _gate_info(fail_metrics, passed=False)
    spec = MethodSpec(key="bandpass", name="带通滤波", params={"low_hz": 8.0},
                      rationale="保守尝试", targets=["低频"], strength=0.5)
    attempt = MethodResult(
        spec=spec,
        metrics=QualityMetrics(**{k: v for k, v in fail_metrics.items()
                                  if k in QualityMetrics.__dataclass_fields__}),
        gate=GateResult(
            passed=False, score=45.0,
            checks=[GateCheck(**{k: v for k, v in c.items()
                                 if k in GateCheck.__dataclass_fields__})
                    for c in fail_gate["checks"]],
            failures=fail_gate["failures"],
            confidence=0.35,
            advice=["降低强度后重试。"],
        ),
        runtime_s=1.2,
        warnings=["信号泄漏偏高。"],
    ).to_info()
    state = {
        "version": "1.0", "job_id": job.name, "input": str(src),
        "stage": "done", "created_at": now_iso(),
        "dataset": ds.to_info(),
        "source": _assessment().to_info(),
        "plan": [spec.to_info(), {"key": "weird", "name": "未知方法"}],  # 故意缺字段
        "attempts": [attempt, {"method": {"key": "partial"}}],  # 故意缺字段
        "best": attempt,
        "gate": fail_gate,
        "confidence": 0.35,
        "history": [],
        "warnings": ["未达到交付标准。"],
    }
    report = build_report(job, state)
    assert report.exists() and report.stat().st_size > 20 * 1024
    doc = Document(str(report))
    text = _docx_text(doc)
    count = text.count("未通过行业门禁")
    assert count >= 3, f"“未通过行业门禁”只出现 {count} 次"
    assert "未达标" in text
    print(f"[docx] 未达标报告 OK，“未通过行业门禁”出现 {count} 次")


def test_clean_report() -> None:
    """干净数据：生成“无需去噪”报告，不走去噪门禁。"""
    clean, noisy, _ = _synthetic()
    _ = clean
    jobs_root = OUT / "jobs_clean"
    if jobs_root.exists():
        shutil.rmtree(jobs_root)
    src = OUT / "输入" / "合成地震记录.segy"
    job = pkg.create_job_folder(src, root=jobs_root)
    ds = _make_dataset(noisy, src)
    ds.save_cache(job / "02_读取与诊断")
    source = SourceAssessment(
        domain="陆地反射地震", acquisition="可控震源", geometry_type="二维",
        confidence=0.92,
        evidence=["信噪比高，无明显面波。"],
        geometry={"n_traces": NTR, "dz_m": DZ},
        cleanliness={"dead_traces": 0},
        noise_types={"随机噪声": 0.1},
        needs_denoise=False,
        recommended=["保持原始数据"],
        clean_reasons=["原始剖面信噪比高，无可见面波与工频干扰。"],
    ).to_info()
    state = {
        "version": "1.0", "job_id": job.name, "input": str(src),
        "stage": "done", "created_at": now_iso(),
        "dataset": ds.to_info(), "source": source,
        "plan": [], "attempts": [], "best": None,
        "gate": None, "confidence": 0.9,
    }
    report = build_report(job, state)
    assert report.exists() and report.stat().st_size > 20 * 1024
    doc = Document(str(report))
    text = _docx_text(doc)
    assert "无需去噪" in text
    assert "未通过行业门禁" not in text
    print(f"[docx] 无需去噪报告 OK（{report.stat().st_size / 1024:.0f} KB）")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    test_figures()
    test_package_and_report()
    test_fail_report()
    test_clean_report()
    elapsed = time.time() - started
    print(f"\n全部通过：{elapsed:.1f}s；产物目录：{OUT}")
    for path in sorted(OUT.glob("jobs/*/05_报告/*.docx")):
        print(f"示例报告：{path}（{path.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
