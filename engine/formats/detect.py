"""格式识别：只读文件头与前若干字节，不加载整份数据。

优先级：强魔数（NPY / HDF5 / MATLAB v5 / ZIP）→ SEG-Y 头段试探
→ 扩展名兜底 → unknown。扩展名只在魔数不成立时使用，避免“改个后缀”骗过识别。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from .segy import BASE_SIZE, probe_segy

_NPY_MAGIC = b"\x93NUMPY"
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_ZIP_MAGIC = b"PK\x03\x04"
_MAT_MAGIC = b"MATLAB 5.0 MAT-file"
_SEGY_SUFFIXES = {".segy", ".sgy", ".segy.gz", ".seg-y"}
_RAW_SUFFIXES = {".raw", ".bin", ".dat", ".d", ".dmp"}
_MAT_SUFFIXES = {".mat"}
_HDF5_SUFFIXES = {".h5", ".hdf5", ".hdf"}
_NPY_SUFFIXES = {".npy"}
_NPZ_SUFFIXES = {".npz"}

# SEG-Y 头段试探低于该分数时视作“不像 SEG-Y”
_SEGY_MIN_SCORE = 0.55


def _zip_has_npy(path: Path) -> bool:
    """ZIP 里是否含 .npy（即 npz 而不是普通压缩包/office 文件）。"""
    try:
        with zipfile.ZipFile(path) as zf:
            return any(name.lower().endswith(".npy") for name in zf.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def _segy_score(path: Path, size: int) -> float:
    """SEG-Y 头段试探分数；失败返回 0。"""
    if size < BASE_SIZE:
        return 0.0
    try:
        return float(probe_segy(path).get("score", 0.0))
    except (ValueError, OSError, KeyError, IndexError, TypeError):
        return 0.0


def detect_format(path: str | Path) -> str:
    """识别文件格式，返回 ``segy/mat/npy/npz/hdf5/raw/unknown`` 之一。"""
    p = Path(path)
    try:
        if not p.is_file():
            return "unknown"
        size = int(p.stat().st_size)
    except OSError:
        return "unknown"
    if size <= 0:
        return "unknown"
    with p.open("rb") as f:
        head = f.read(8192)
    suffix = p.suffix.lower()

    # ---- 强魔数 ----
    for offset in (0, 512, 1024, 2048, 4096):
        if head[offset:offset + len(_HDF5_MAGIC)] == _HDF5_MAGIC:
            # MATLAB v7.3 文件本质是 HDF5，后缀能区分
            return "mat" if suffix in _MAT_SUFFIXES else "hdf5"
    if head.startswith(_NPY_MAGIC):
        return "npy"
    if head.startswith(_ZIP_MAGIC):
        if _zip_has_npy(p):
            return "npz"
        return "unknown"
    if head.startswith(_MAT_MAGIC) or head[128:128 + len(_MAT_MAGIC)] == _MAT_MAGIC:
        return "mat"

    # ---- SEG-Y 头段试探（需要 3600 字节以上）----
    if size >= BASE_SIZE and _segy_score(p, size) >= _SEGY_MIN_SCORE:
        return "segy"

    # ---- 扩展名兜底 ----
    if suffix in _RAW_SUFFIXES:
        return "raw"
    if suffix in _SEGY_SUFFIXES:
        return "segy"
    if suffix in _MAT_SUFFIXES:
        return "mat"
    if suffix in _NPY_SUFFIXES:
        return "npy"
    if suffix in _NPZ_SUFFIXES:
        return "npz"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    return "unknown"
