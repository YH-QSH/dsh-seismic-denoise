"""通用小工具：随机种子、日志、JSON、计时。"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


# --------------------------------------------------------------------------
# 随机种子
# --------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """尽可能固定所有随机源，保证可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 关闭 cudnn 自动调优，保证同一 seed 结果一致
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
def setup_console_encoding() -> None:
    """把 stdout/stderr 统一成 UTF-8。

    Windows 控制台默认 cp936，而终端往往按 UTF-8 解码，中文会变乱码；
    reload 成 UTF-8 之后 Python 的输出就和终端一致了。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = (getattr(stream, "encoding", "") or "").lower()
            if enc not in ("utf-8", "utf8") and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def get_logger(name: str = "vsdn", logfile: str | Path | None = None) -> logging.Logger:
    setup_console_encoding()
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加 handler
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class Timer:
    """with Timer('训练'): ...  —— 打印耗时。"""

    def __init__(self, name: str = "", logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        msg = f"{self.name} 耗时 {dt:.1f} s"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)
        return False


# --------------------------------------------------------------------------
# JSON / 目录
# --------------------------------------------------------------------------
def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"不可序列化类型: {type(o)}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# 数组小工具
# --------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    """均方根。"""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def to_float32(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def describe_array(name: str, a: np.ndarray) -> str:
    a = np.asarray(a)
    return (
        f"{name}: shape={a.shape} dtype={a.dtype} "
        f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} "
        f"rms={rms(a):.4g} nan={int(np.isnan(a).sum())}"
    )
