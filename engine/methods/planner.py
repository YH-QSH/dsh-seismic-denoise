"""方法规划器：根据来源评估与历史尝试，给出本轮要跑的去噪方法。

规则（见 docs/接口约定.md）：
- 按 ``assessment.noise_types`` 的评分从高到低命中；
- 面波 → fk / bandpass / median；随机噪声 → fx_decon / spectral_sub / cnn；
- 工频干扰 → notch；散射绕射 → fk / svd / cnn；折射波 → fk / median；
- 未知 → 保守 bandpass + cnn；每轮最多 4 个方案。

``params`` 与 ``engine/methods/classical.py``、``cnn.py`` 的默认参数同名字段一致，
可直接传给 ``run_method`` / ``run_cnn``。
"""
from __future__ import annotations

import copy
import json
import math

from ..model import MethodSpec

# --------------------------------------------------------------------------
# 方法元信息（与 classical.available_methods() 保持一致，规划器不 import 方法模块）
# --------------------------------------------------------------------------
_METHOD_NAMES: dict[str, str] = {
    "bandpass": "带通滤波",
    "notch": "工频陷波",
    "despike": "异常振幅压制",
    "median": "中值滤波",
    "fk": "F-K 视速度滤波",
    "fx_decon": "F-X 预测滤波",
    "svd": "SVD 秩截断",
    "spectral_sub": "谱减法",
    "wavelet": "小波阈值去噪",
    "cnn": "CNN 分类分离",
}

_METHOD_TARGETS: dict[str, list[str]] = {
    "bandpass": ["面波", "随机噪声", "工频干扰"],
    "notch": ["工频干扰"],
    "despike": ["随机噪声"],
    "median": ["面波", "随机噪声"],
    "fk": ["面波", "折射波", "散射绕射"],
    "fx_decon": ["随机噪声", "散射绕射"],
    "svd": ["散射绕射", "面波"],
    "spectral_sub": ["随机噪声", "工频干扰"],
    "wavelet": ["随机噪声"],
    "cnn": ["面波", "随机噪声", "工频干扰", "散射绕射", "折射波"],
}

# 默认参数：保守起步，避免第一轮就把有效波削光
_DEFAULT_PARAMS: dict[str, dict] = {
    "bandpass": {"f_lo": 5.0, "f_hi": 90.0, "order": 4},
    "notch": {"freq": 50.0, "harmonics": 3, "q": 30.0},
    "despike": {"window": 5, "threshold": 6.0},
    "median": {"kt": 5, "kx": 1},
    "fk": {"vmin": 1200.0, "taper": 400.0, "reject_slow": True,
           "f_lo": 2.0, "f_hi": None},
    "fx_decon": {"win": 9, "lags": 2, "alpha": 0.85, "eps": 0.001,
                 "f_lo": 4.0, "f_hi": 120.0},
    "svd": {"wt_ms": 256.0, "wx": 16, "rank": 2, "step": 0.5},
    "spectral_sub": {"nperseg_ms": 128.0, "overlap": 0.75, "alpha": 1.5,
                     "floor": 0.05, "noise_percentile": 20.0,
                     "max_reduction_db": 20.0},
    "wavelet": {"level": 4, "wavelet": "sym8", "threshold_scale": 1.0},
    "cnn": {"alpha": 0.65, "gate": True, "device": "auto", "resample": False},
}

# 噪声类型 → 方法优先级（顺序即优先级）
_NOISE_RULES: dict[str, list[str]] = {
    "面波": ["fk", "bandpass", "median"],
    "随机噪声": ["fx_decon", "spectral_sub", "cnn"],
    "工频干扰": ["notch"],
    "散射绕射": ["fk", "svd", "cnn"],
    "折射波": ["fk", "median"],
}

# 噪声名同义词 → 标准噪声类型
_NOISE_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("面波", ("面波", "瑞雷", "rayleigh", "groundroll", "ground roll",
              "surface wave", "surface-wave", "线性噪声", "低速")),
    ("随机噪声", ("随机", "random", "背景噪声", "高频噪声")),
    ("工频干扰", ("工频", "powerline", "power line", "50hz", "50 hz", "60hz",
                  "60 hz", "陷波", "mains")),
    ("散射绕射", ("散射", "绕射", "scatter", "diffract")),
    ("折射波", ("折射", "refraction", "refracted")),
]

_KIND_LABEL = {
    "over": "伤及有效波（泄漏/能量/频谱不达标）",
    "under": "去噪不足（信噪比增益不够）",
    "noop": "近乎原样返回",
    "error": "执行失败",
    "other": "门禁未通过",
}


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _cfg_dict(cfg) -> dict:
    return cfg if isinstance(cfg, dict) else {}


def _planner_cfg(cfg: dict) -> dict:
    p = cfg.get("planner")
    return p if isinstance(p, dict) else {}


def _canonical_noise(name) -> str:
    s = str(name or "").strip()
    low = s.lower()
    for canon, aliases in _NOISE_ALIASES:
        for a in aliases:
            if a in low:
                return canon
    return s


def _freeze(params: dict) -> str:
    def _norm(v):
        if isinstance(v, float):
            return round(v, 6)
        return v
    try:
        obj = {k: _norm(v) for k, v in (params or {}).items()}
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(sorted((params or {}).items()))


def _method_params(key: str, cfg: dict, extra: dict | None = None) -> dict:
    """默认参数 + 用户/配置覆盖 + 本轮额外覆盖。"""
    base = copy.deepcopy(_DEFAULT_PARAMS.get(key, {}))
    cfg_override = _cfg_dict(cfg)
    # 允许 cfg["method_params"] / cfg["params"] / cfg["planner"]["method_params"]
    for src in (cfg_override.get("method_params"), cfg_override.get("params"),
                _planner_cfg(cfg_override).get("method_params")):
        if isinstance(src, dict) and isinstance(src.get(key), dict):
            base.update(src[key])
    if key == "cnn":
        # 流水线/命令行把 CNN 参数放在 cfg["cnn"]
        cnn_cfg = cfg_override.get("cnn")
        if isinstance(cnn_cfg, dict):
            base.update(cnn_cfg)
        if cfg_override.get("device"):
            base["device"] = cfg_override["device"]
    if extra:
        base.update(extra)
    return base


def _filter_keys(keys: list[str], cfg: dict, planner: dict) -> list[str]:
    allow = planner.get("allow")
    if isinstance(allow, (list, tuple, set)):
        allowed = {str(k) for k in allow}
        keys = [k for k in keys if k in allowed]
    deny = planner.get("deny")
    if isinstance(deny, (list, tuple, set)):
        denied = {str(k) for k in deny}
        keys = [k for k in keys if k not in denied]
    allow_cnn = planner.get("allow_cnn", cfg.get("allow_cnn", True))
    if allow_cnn is False:
        keys = [k for k in keys if k != "cnn"]
    return keys


def _noise_rank(assessment, planner: dict) -> list[tuple[str, float, list[str]]]:
    """返回 [(标准噪声名, 评分, 原始名列表)]，按评分降序。"""
    raw = getattr(assessment, "noise_types", None) or {}
    if not isinstance(raw, dict):
        return []
    min_score = float(planner.get("min_noise_score", 0.15))
    items: dict[str, float] = {}
    origins: dict[str, list[str]] = {}
    for k, v in raw.items():
        try:
            s = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(s) or s <= 0:
            continue
        canon = _canonical_noise(k)
        items[canon] = max(items.get(canon, 0.0), s)
        origins.setdefault(canon, []).append(str(k))
    if items and max(items.values()) > 2.0:  # 百分制 → 0–1
        items = {k: v / 100.0 for k, v in items.items()}
    order = {name: i for i, name in enumerate(_NOISE_RULES)}
    ranked = [(k, v, origins.get(k, [k])) for k, v in items.items() if v >= min_score]
    ranked.sort(key=lambda t: (-t[1], order.get(t[0], 99)))
    return ranked


def _explicit_specs(cfg: dict) -> list[MethodSpec] | None:
    """支持 cfg 里直接给方法列表（字符串 key / dict / MethodSpec）。"""
    raw = cfg.get("methods")
    if raw is None and isinstance(cfg.get("plan"), list):
        raw = cfg["plan"]
    if raw is None or isinstance(raw, (str, bytes)):
        if isinstance(raw, str) and raw.strip():
            raw = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            return None
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    specs: list[MethodSpec] = []
    for item in raw:
        if isinstance(item, MethodSpec):
            specs.append(copy.deepcopy(item))
        elif isinstance(item, dict) and item.get("key"):
            key = str(item["key"])
            specs.append(MethodSpec(
                key=key,
                name=str(item.get("name") or _METHOD_NAMES.get(key, key)),
                params=dict(item.get("params") or {}),
                rationale=str(item.get("rationale") or "用户指定"),
                targets=list(item.get("targets") or _METHOD_TARGETS.get(key, [])),
                strength=float(item.get("strength", 1.0)),
                priority=int(item.get("priority", len(specs))),
                source=str(item.get("source") or "user"),
            ))
        else:
            key = str(item)
            specs.append(MethodSpec(
                key=key, name=_METHOD_NAMES.get(key, key),
                params=_method_params(key, cfg), rationale="用户指定",
                targets=_METHOD_TARGETS.get(key, []), source="user",
            ))
    return specs


# --------------------------------------------------------------------------
# 规划
# --------------------------------------------------------------------------
def plan(assessment, cfg: dict | None = None) -> list[MethodSpec]:
    """给出一轮去噪方案（主方法在前，最多 4 个）。"""
    cfg = _cfg_dict(cfg)
    planner = _planner_cfg(cfg)
    max_methods = int(planner.get("max_methods", 4) or 4)
    max_methods = max(1, min(4, max_methods))

    # 用户显式指定：原样采用（也做数量上限）
    explicit = _explicit_specs(cfg)
    if explicit is not None:
        return explicit[:max_methods]

    # 评估认为无需去噪：不生成方案
    if getattr(assessment, "needs_denoise", True) is False:
        return []

    ranked = _noise_rank(assessment, planner)
    recommended = list(getattr(assessment, "recommended", None) or [])
    use_rec = bool(planner.get("use_recommended", True))

    keys: list[str] = []
    if use_rec:
        for k in recommended:
            k = str(k)
            if k in _DEFAULT_PARAMS and k not in keys:
                keys.append(k)
    for canon, _score, _origins in ranked:
        for k in _NOISE_RULES.get(canon, []):
            if k not in keys:
                keys.append(k)
    if not keys:
        # 未知噪声：保守 bandpass + CNN
        keys = ["bandpass", "cnn"]
        fallback = True
    else:
        fallback = False

    keys = _filter_keys(keys, cfg, planner)
    if not keys:
        # 过滤后空了，退回最保守方案
        keys = _filter_keys(["bandpass", "cnn"], cfg, planner)
    keys = keys[:max_methods]

    dominant = ranked[0] if ranked else (None, 0.0, [])
    specs: list[MethodSpec] = []
    for i, key in enumerate(keys):
        detected = {c for c, _s, _o in ranked}
        targets = [t for t in _METHOD_TARGETS.get(key, []) if t in detected] \
            or list(_METHOD_TARGETS.get(key, []))
        if i == 0 and dominant[0]:
            why = (f"主噪声为{dominant[0]}（评分 {dominant[1]:.2f}），"
                   f"首选{_METHOD_NAMES.get(key, key)}；参数保守起步")
        elif fallback and key == "bandpass":
            why = "未识别出明确噪声类型，先做保守带通再观察"
        elif fallback and key == "cnn":
            why = "未识别出明确噪声类型，用 CNN 分类分离做数据驱动兜底"
        else:
            why = f"备选方案，压制{'、'.join(targets) or '剩余噪声'}"
        specs.append(MethodSpec(
            key=key,
            name=_METHOD_NAMES.get(key, key),
            params=_method_params(key, cfg),
            rationale=why,
            targets=targets,
            strength=round(max(0.4, 1.0 - 0.12 * i), 2),
            priority=i,
            source="rule",
        ))
    return specs


# --------------------------------------------------------------------------
# 失败原因与参数调整
# --------------------------------------------------------------------------
def _gate_thresholds(cfg: dict) -> dict[str, float]:
    """与 quality/gate.py 对齐的默认阈值（planner 不 import 门禁模块）。"""
    th = {
        "min_gain_db": 3.0,
        "energy_ratio_min": 0.60,
        "energy_ratio_max": 1.05,
        "signal_leakage_max": 0.40,
        "spectrum_corr_min": 0.80,
        "clip_ratio_max": 1e-3,
        "snr_out_min_db": 10.0,
        "identity_gain_db": 0.2,
        "identity_energy": 0.03,
    }
    gate_cfg = cfg.get("gate")
    if isinstance(gate_cfg, dict):
        for k, v in gate_cfg.items():
            if k in th:
                try:
                    th[k] = float(v)
                except (TypeError, ValueError):
                    continue
    return th


def _val(metrics, name: str):
    if metrics is None:
        return None
    v = getattr(metrics, name, None)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _failure_kind(result, cfg: dict) -> str:
    """把一次失败归类：over（伤信号）/ under（去噪不足）/ noop / error / other。"""
    gate = getattr(result, "gate", None)
    metrics = getattr(result, "metrics", None)
    warnings = " ".join(str(w) for w in (getattr(result, "warnings", None) or []))
    failures = " ".join(str(f) for f in (getattr(gate, "failures", None) or []))
    if "执行失败" in failures or "推理失败" in failures or "不可用" in warnings:
        return "error"

    th = _gate_thresholds(cfg)
    gain = _val(metrics, "snr_gain_db")
    energy = _val(metrics, "energy_ratio")
    leak = _val(metrics, "signal_leakage")
    spec = _val(metrics, "spectrum_corr")
    clip = _val(metrics, "clip_ratio")
    if gain is None and energy is None:
        return "error" if failures else "other"

    if (leak is not None and leak > th["signal_leakage_max"]) \
            or (energy is not None and energy < th["energy_ratio_min"]) \
            or (energy is not None and energy > th["energy_ratio_max"]) \
            or (spec is not None and spec < th["spectrum_corr_min"]) \
            or (clip is not None and clip > th["clip_ratio_max"]):
        return "over"
    if gain is not None and abs(gain) < th["identity_gain_db"] \
            and energy is not None and abs(energy - 1.0) < th["identity_energy"]:
        return "noop"
    if gain is None or gain < th["min_gain_db"]:
        return "under"
    return "other"


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _adjust(key: str, params: dict, kind: str) -> dict:
    """按失败类型调整参数：over 降强度，under 加强度，noop 处理空转。"""
    p = dict(params or {})
    if kind in ("error",):
        if key == "cnn":
            p["device"] = "cpu"
        return p

    if key == "cnn":
        alpha = _clip(float(p.get("alpha", 0.65)), 0.05, 0.98)
        if kind == "noop":
            p["gate"] = False          # 分类概率全低于阈值时，关掉门控再试
            p["alpha"] = _clip(alpha * 1.15, 0.05, 0.9)
        elif kind == "over":
            p["alpha"] = round(_clip(alpha * 0.6, 0.1, 0.95), 3)
        else:
            p["alpha"] = round(_clip(alpha * 1.25, 0.1, 0.95), 3)
        return p

    if key == "bandpass":
        f_lo = float(p.get("f_lo", 5.0))
        f_hi = float(p.get("f_hi", 90.0))
        order = int(p.get("order", 4))
        if kind == "over":
            p["f_lo"] = round(max(1.0, f_lo * 0.7), 2)
            p["f_hi"] = round(min(200.0, f_hi * 1.2), 2)
            p["order"] = max(2, order - 1)
        else:
            p["f_lo"] = round(min(30.0, f_lo * 1.25), 2)
            p["f_hi"] = round(max(20.0, f_hi * 0.85), 2)
            p["order"] = min(8, order + 2)
        return p

    if key == "notch":
        h = int(p.get("harmonics", 3))
        q = float(p.get("q", 30.0))
        if kind == "over":
            p["harmonics"] = max(1, h - 1)
            p["q"] = round(min(120.0, q * 1.5), 2)
        else:
            p["harmonics"] = min(6, h + 1)
            p["q"] = round(max(10.0, q * 0.8), 2)
        return p

    if key == "median":
        kt = int(p.get("kt", 5))
        p["kt"] = max(3, kt - 2) if kind == "over" else min(15, kt + 2)
        return p

    if key == "fk":
        vmin = float(p.get("vmin", 1200.0))
        taper = float(p.get("taper", 400.0))
        if kind == "over":
            p["vmin"] = round(max(200.0, vmin * 0.8), 1)
            p["taper"] = round(max(100.0, taper * 1.2), 1)
        else:
            p["vmin"] = round(min(6000.0, vmin * 1.25), 1)
            p["taper"] = round(max(100.0, taper * 0.9), 1)
        return p

    if key == "fx_decon":
        alpha = _clip(float(p.get("alpha", 0.85)), 0.05, 0.99)
        win = int(p.get("win", 9))
        if kind == "over":
            p["alpha"] = round(_clip(alpha * 0.65, 0.1, 0.9), 3)
            p["win"] = max(5, win - 2)
        else:
            p["alpha"] = round(_clip(alpha * 1.1, 0.1, 0.98), 3)
            p["win"] = min(15, win + 2)
        p["win"] = int(p["win"])
        return p

    if key == "svd":
        rank = int(p.get("rank", 2))
        wx = int(p.get("wx", 16))
        if kind == "over":
            p["rank"] = min(8, rank + 1)     # 保留更多奇异值，少削信号
        else:
            p["rank"] = max(1, rank - 1)     # 只留最强相干成分
            p["wx"] = min(32, wx + 8)
        return p

    if key == "spectral_sub":
        alpha = float(p.get("alpha", 1.5))
        floor = float(p.get("floor", 0.05))
        if kind == "over":
            p["alpha"] = round(max(0.5, alpha * 0.6), 2)
            p["floor"] = round(min(0.3, floor * 1.6), 3)
        else:
            p["alpha"] = round(min(4.0, alpha * 1.3), 2)
            p["floor"] = round(max(0.01, floor * 0.7), 3)
        return p

    if key == "wavelet":
        scale = float(p.get("threshold_scale", 1.0))
        p["threshold_scale"] = round(
            max(0.2, scale * 0.6) if kind == "over" else min(3.0, scale * 1.4), 3)
        return p

    if key == "despike":
        thr = float(p.get("threshold", 6.0))
        p["threshold"] = round(
            min(15.0, thr * 1.4) if kind == "over" else max(3.0, thr * 0.8), 2)
        return p

    return p


def refine(assessment, history: list, cfg: dict | None = None) -> list[MethodSpec]:
    """根据上一轮结果给出新方案：优先新方法，其次按失败原因调参重试。"""
    cfg = _cfg_dict(cfg)
    planner = _planner_cfg(cfg)
    max_methods = int(planner.get("max_methods", 4) or 4)
    max_methods = max(1, min(4, max_methods))
    max_total = int(planner.get("max_total_attempts", cfg.get("max_attempts", 10)) or 10)

    history = [r for r in (history or []) if getattr(r, "spec", None) is not None]
    if len(history) >= max_total:
        return []
    if getattr(assessment, "needs_denoise", True) is False and not history:
        return []

    retry_failed = bool(planner.get("retry_failed", True))
    passed_keys: set[str] = set()
    failed: list[tuple[str, dict, str, object]] = []
    seen_pairs: set[str] = set()
    for r in history:
        spec = r.spec
        key = str(spec.key)
        params = dict(spec.params or {})
        seen_pairs.add(_freeze({"key": key, "params": params}))
        gate = getattr(r, "gate", None)
        if gate is not None and getattr(gate, "passed", False):
            passed_keys.add(key)
        else:
            failed.append((key, params, _failure_kind(r, cfg), r))

    tried_keys = {f[0] for f in failed} | passed_keys
    candidates: list[MethodSpec] = []
    emitted: set[str] = set()

    def _push(spec: MethodSpec) -> None:
        fp = _freeze({"key": spec.key, "params": spec.params})
        if fp in seen_pairs or fp in emitted or spec.key in passed_keys:
            return
        emitted.add(fp)
        candidates.append(spec)

    # ---- 1) 全新方法：用 plan 的规则顺序，但排除已试过/已通过的 key ----
    base = plan(assessment, cfg)
    for spec in base:
        if spec.key in tried_keys:
            continue
        _push(spec)

    # ---- 1b) 高保真兜底：带通先保住宽频有效波，再决定是否加码 ----
    if "bandpass" not in tried_keys and all(s.key != "bandpass" for s in candidates):
        _push(MethodSpec(
            key="bandpass",
            name=_METHOD_NAMES.get("bandpass", "bandpass"),
            params=_method_params("bandpass", cfg),
            rationale="高保真兜底：先用保守带通保住宽频有效波与弱同相轴，再比较其他方法",
            targets=list(_METHOD_TARGETS.get("bandpass", [])),
            priority=len(candidates),
            source="fallback",
        ))

    # ---- 2) 对失败方法按原因调整参数后重试 ----
    if retry_failed:
        for key, params, kind, result in failed:
            if kind == "error" and key != "cnn":
                continue
            if kind == "noop" and key != "cnn":
                continue
            newp = _adjust(key, params, kind)
            if _freeze(newp) == _freeze(params):
                continue
            label = _KIND_LABEL.get(kind, kind)
            _push(MethodSpec(
                key=key,
                name=_METHOD_NAMES.get(key, key),
                params=newp,
                rationale=f"上一轮{_METHOD_NAMES.get(key, key)}未通过（{label}），"
                          f"{'降低强度/加保护' if kind == 'over' else '调整参数加强'}后重试",
                targets=list(_METHOD_TARGETS.get(key, [])),
                strength=1.0 if kind != "over" else 0.6,
                priority=len(candidates),
                source="refine",
            ))
            if len(candidates) >= max_methods:
                break

    return candidates[:max_methods]
