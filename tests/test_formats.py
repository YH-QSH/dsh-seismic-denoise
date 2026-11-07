"""formats 层自测：真实 Whataroa 数据 + 合成 SEG-Y / 容器 / RAW。

运行方式（pytest 未安装时直接跑脚本）：
    python tests/test_formats.py
或（装了 pytest）：
    python -m pytest tests/test_formats.py -q

覆盖：
1. 真实 Whataroa SEG-Y：格式识别、ns/ntr/dt/dz、IBM float、EBCDIC 文本头、道头字段；
2. 合成 IEEE float / IBM float SEG-Y 写读回环（误差 < 1e-5）与道头 1-based 解析；
3. .npy / .npz / .mat / .hdf5 / MAT v7.3 小样本，read_any 都能读；
4. RAW 候选扫描与读取（不崩、低置信度、有明确警告）；
5. analyze_binary 对 SEG-Y 与未知二进制的初判；
6. max_traces 截断警告与形状。

所有产物写到 tests/_out/formats/，每次运行先清空，可重复跑。
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.formats import (  # noqa: E402
    analyze_binary,
    detect_format,
    read_any,
    read_raw,
    read_segy,
    read_segy_headers,
    write_segy,
)
from engine.model import HeaderTable  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out" / "formats"
_REAL_ENV = os.environ.get("SEISMIC_REAL_DATA_DIR")
REAL_DIR: Path | None = Path(_REAL_ENV) if _REAL_ENV else None
REAL_NAMES = (
    "whataroa_0_hDVS_2239.segy",
    "whataroa_1_hDVS_6101.segy",
    "whataroa_2_hDVS_8125.segy",
)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _prepare_out() -> Path:
    """每次运行重建输出目录，保证可重复。"""
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _synthetic_data(ns: int = 400, ntr: int = 8, seed: int = 7) -> np.ndarray:
    """带横向连续性的合成地震道集：衰减正弦 + 小噪声。"""
    rng = np.random.default_rng(seed)
    t = np.arange(ns, dtype=np.float32)[:, None]
    r = np.arange(ntr, dtype=np.float32)[None, :]
    data = np.sin(2.0 * np.pi * t / 37.0 + 0.3 * r) * np.exp(-t / 500.0)
    data = data + 0.05 * rng.standard_normal((ns, ntr)).astype(np.float32)
    return data.astype(np.float32)


def _synthetic_headers(ntr: int, *, dt_us: int = 2000) -> HeaderTable:
    """造一张 1-based 道头表：每道高程差 205（配合 -100 标定 → 2.05 m）。"""
    table = HeaderTable()
    idx = np.arange(ntr, dtype=np.float64)
    table.add("trace_seq", "道序号", 1, 4, "int", idx + 1.0)
    table.add("energy_source", "震源点号", 9, 4, "int", idx * 10.0 + 100.0)
    table.add("shotpoint", "炮点号", 17, 4, "int", np.full(ntr, 1.0))
    table.add("offset", "炮检距", 37, 4, "int", 100.0 + 25.0 * idx)
    table.add("receiver_elev", "接收点高程", 41, 4, "int", 1000.0 + 205.0 * idx)
    table.add("source_elev", "炮点高程", 45, 4, "int", np.full(ntr, 1200.0))
    table.add("source_depth", "炮点深度", 49, 4, "int", np.zeros(ntr))
    table.add("receiver_depth", "接收点深度", 53, 4, "int", np.zeros(ntr))
    table.add("elev_scalar", "高程标定因子", 69, 2, "int", np.full(ntr, -100.0))
    table.add("coord_scalar", "坐标标定因子", 71, 2, "int", np.full(ntr, -100.0))
    table.add("source_x", "炮点 X", 73, 4, "int", np.full(ntr, 500000.0))
    table.add("source_y", "炮点 Y", 77, 4, "int", np.full(ntr, 3000000.0))
    table.add("group_x", "检波点 X", 81, 4, "int", 500000.0 + 2500.0 * idx)
    table.add("group_y", "检波点 Y", 85, 4, "int", np.full(ntr, 3000000.0))
    table.add("coord_unit", "坐标单位", 89, 2, "int", np.full(ntr, 1.0))
    table.add("ns", "每道采样点数", 115, 2, "int", np.full(ntr, 400.0))
    table.add("dt", "采样间隔 us", 117, 2, "int", np.full(ntr, float(dt_us)))
    return table


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return a.shape == b.shape and float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) <= tol


def _real_files() -> list[Path]:
    if REAL_DIR is None or not REAL_DIR.is_dir():
        return []
    return [REAL_DIR / name for name in REAL_NAMES if (REAL_DIR / name).is_file()]


# --------------------------------------------------------------------------
# 1. 真实 SEG-Y
# --------------------------------------------------------------------------
def test_real_whataroa():
    files = _real_files()
    if len(files) < len(REAL_NAMES):
        print(f"  [跳过] 未找到真实数据：{REAL_DIR or '未设置 SEISMIC_REAL_DATA_DIR'}")
        return
    assert detect_format(files[0]) == "segy"
    for path in files:
        ds = read_any(path)
        assert ds.format == "segy", f"{path.name}: format={ds.format}"
        assert ds.source_kind == "segy"
        assert ds.data.dtype == np.float32
        assert ds.data.shape == (4000, 436), f"{path.name}: shape={ds.data.shape}"
        assert ds.n_samples == 4000 and ds.n_traces == 436
        assert ds.dt is not None and abs(ds.dt - 0.001) < 1e-9, f"{path.name}: dt={ds.dt}"
        assert ds.dz is not None and abs(ds.dz - 2.05) < 1e-6, f"{path.name}: dz={ds.dz}"
        assert np.isfinite(ds.data).all()
        assert float(np.max(np.abs(ds.data))) < 1e6
        assert ds.confidence > 0.8, f"{path.name}: confidence={ds.confidence}"
        assert not any("截断" in w for w in ds.warnings)
        assert ds.meta["endian"] == "big"
        assert ds.meta["sample_format"] == 1
        assert ds.meta["n_traces_read"] == 436
        # 文本头应为 EBCDIC（cp500）解出来的可读英文
        assert ds.meta["text_encoding"] == "cp500", ds.meta["text_encoding"]
        assert "WIRELINE" in ds.text_header.upper(), ds.text_header[:80]
        assert len(ds.headers.fields) >= 20

    # 只读道头：1-based 字段值与样本对齐
    table = read_segy_headers(files[0])
    assert len(table.values("trace_seq")) == 436
    assert table.get("ns").byte == 115 and table.get("ns").size == 2
    assert table.get("dt").byte == 117
    assert table.get("receiver_elev").byte == 41
    assert np.all(table.values("ns") == 4000)
    assert np.all(table.values("dt") == 1000)
    elev = table.values("receiver_elev")
    assert elev[0] == 9461 and elev[1] == 9256


def test_real_max_traces():
    files = _real_files()
    if not files:
        print("  [跳过] 无真实数据")
        return
    ds = read_any(files[0], max_traces=50)
    assert ds.data.shape == (4000, 50)
    assert any("max_traces" in w for w in ds.warnings), ds.warnings
    assert analyze_binary(files[0])["format"] == "segy"


# --------------------------------------------------------------------------
# 2. 合成 SEG-Y 回环（IEEE / IBM）
# --------------------------------------------------------------------------
def _roundtrip_case(name: str, fmt: int) -> None:
    out = _prepare_out()
    ns, ntr, dt = 400, 8, 0.002
    data = _synthetic_data(ns, ntr)
    headers = _synthetic_headers(ntr, dt_us=2000)
    path = out / f"{name}.segy"
    write_segy(path, data, dt, fmt=fmt, text="C01 SYNTHETIC TEST\nC40 END", headers=headers)

    assert detect_format(path) == "segy"
    ds = read_any(path)
    assert ds.format == "segy" and ds.data.shape == (ns, ntr)
    assert ds.dt is not None and abs(ds.dt - dt) < 1e-12, ds.dt
    tol = 1e-5 if fmt == 1 else 1e-6
    err = float(np.max(np.abs(ds.data.astype(np.float64) - data.astype(np.float64))))
    assert err < tol, f"{name}: 回环误差 {err}"
    assert ds.dz is not None and abs(ds.dz - 2.05) < 1e-6, f"{name}: dz={ds.dz}"
    assert ds.meta["sample_format"] == fmt
    assert ds.confidence > 0.6, f"{name}: confidence={ds.confidence}"
    assert "SYNTHETIC TEST" in ds.text_header

    # 道头 1-based 解析
    table = read_segy_headers(path)
    assert np.array_equal(table.values("trace_seq"), np.arange(1, ntr + 1))
    assert np.all(table.values("ns") == ns)
    assert np.all(table.values("dt") == 2000)
    field = table.get("receiver_elev")
    assert field.byte == 41 and field.size == 4
    assert np.array_equal(field.values, 1000.0 + 205.0 * np.arange(ntr))

    # 独立 read_segy 与 read_any 一致
    ds2 = read_segy(path)
    assert np.array_equal(ds.data, ds2.data)


def test_synthetic_ieee_roundtrip():
    _roundtrip_case("syn_ieee", fmt=5)


def test_synthetic_ibm_roundtrip():
    _roundtrip_case("syn_ibm", fmt=1)


# --------------------------------------------------------------------------
# 2b. 变长道 SEG-Y（各道 ns 不同）
# --------------------------------------------------------------------------
def test_variable_length_segy():
    import struct

    out = _prepare_out()
    ns_bin, dt_us = 400, 2000
    ns_list = [400, 380, 420]
    text = b" " * 3200
    binh = bytearray(400)
    struct.pack_into(">H", binh, 16, dt_us)     # 二进制头 dt（块内 0-based 16）
    struct.pack_into(">H", binh, 20, ns_bin)    # ns
    struct.pack_into(">H", binh, 24, 5)         # 格式码 5 = IEEE float
    struct.pack_into(">i", binh, 304, 0)        # 扩展文本头数

    blob = bytearray(text + binh)
    for k, ns_k in enumerate(ns_list):
        th = bytearray(240)
        struct.pack_into(">i", th, 0, k + 1)              # 道序号
        struct.pack_into(">i", th, 40, 1000 + 205 * k)    # 接收点高程（原始值）
        struct.pack_into(">h", th, 68, -100)              # 高程标定因子
        struct.pack_into(">h", th, 114, ns_k)             # 本道 ns
        struct.pack_into(">h", th, 116, dt_us)            # 本道 dt
        blob += th
        tr = (np.arange(ns_k, dtype=np.float32) + 1.0)
        blob += np.asarray(tr, dtype=">f4").tobytes()

    path = out / "variable.sgy"
    path.write_bytes(bytes(blob))

    assert detect_format(path) == "segy"
    ds = read_any(path)
    assert ds.data.shape == (ns_bin, len(ns_list)), ds.data.shape
    assert ds.meta["variable_length"] is True
    assert ds.meta["trace_ns_min"] == 380 and ds.meta["trace_ns_max"] == 420
    assert any("变长道" in w for w in ds.warnings), ds.warnings
    # 第 2 道（ns=380）尾部补 0，第 3 道（ns=420）截断到中位数 400
    assert np.allclose(ds.data[:, 0], np.arange(1, 401))
    assert np.allclose(ds.data[:380, 1], np.arange(1, 381))
    assert np.all(ds.data[380:, 1] == 0)
    assert np.allclose(ds.data[:, 2], np.arange(1, 401))
    assert ds.dt is not None and abs(ds.dt - 0.002) < 1e-12
    assert ds.dz is not None and abs(ds.dz - 2.05) < 1e-6
    # 道头逐道解析
    table = read_segy_headers(path)
    assert np.array_equal(table.values("ns"), np.asarray(ns_list, dtype=np.float64))


# --------------------------------------------------------------------------
# 3. 容器格式
# --------------------------------------------------------------------------
def test_npy_npz():
    out = _prepare_out()
    data = _synthetic_data(200, 6)
    npy = out / "sample.npy"
    np.save(npy, data)

    assert detect_format(npy) == "npy"
    ds = read_any(npy)
    assert ds.format == "npy" and ds.source_kind == "npy"
    assert _close(ds.data, data)
    assert ds.dt is None and any("dt" in w for w in ds.warnings)

    npz = out / "sample.npz"
    np.savez(npz, data=data, dt=np.array(0.004), dz=np.array(5.0))
    assert detect_format(npz) == "npz"
    ds = read_any(npz)
    assert ds.format == "npz" and _close(ds.data, data)
    assert ds.dt is not None and abs(ds.dt - 0.004) < 1e-12, ds.dt
    assert ds.dz is not None and abs(ds.dz - 5.0) < 1e-12, ds.dz


def test_mat_and_hdf5():
    import h5py
    import scipy.io

    out = _prepare_out()
    data = _synthetic_data(240, 6)

    # --- MAT v5：一个 struct 变量里含 data + header + trace_headers ---
    mat = out / "sample.mat"
    ntr = data.shape[1]
    scipy.io.savemat(
        mat,
        {
            "seis": {
                "data": data,
                "header": {"dt": np.array([[0.002]])},
                "trace_headers": {
                    "ReceiverGroupElevation": np.asarray(1000.0 + 205.0 * np.arange(ntr)),
                    "ReceiverGroupDepth": np.zeros(ntr),
                },
            },
        },
    )
    assert detect_format(mat) == "mat"
    ds = read_any(mat)
    assert ds.format == "mat", ds.format
    assert _close(ds.data, data)
    assert ds.dt is not None and abs(ds.dt - 0.002) < 1e-12, ds.dt
    assert ds.dz is None or abs(ds.dz - 2.05) < 1e-6, ds.dz
    assert "trace_headers" in ds.meta or ds.headers.fields

    # --- 普通 HDF5 ---
    h5 = out / "sample.h5"
    with h5py.File(str(h5), "w") as f:
        dset = f.create_dataset("data", data=data)
        dset.attrs["dt"] = 0.001
        dset.attrs["dz"] = 2.5
    assert detect_format(h5) == "hdf5"
    ds = read_any(h5)
    assert ds.format == "hdf5" and _close(ds.data, data)
    assert ds.dt is not None and abs(ds.dt - 0.001) < 1e-12, ds.dt
    assert ds.dz is not None and abs(ds.dz - 2.5) < 1e-12, ds.dz

    # --- MAT v7.3（HDF5 内核）走 MAT 入口 ---
    v73 = out / "sample_v73.mat"
    with h5py.File(str(v73), "w") as f:
        dset = f.create_dataset("data", data=data)
        dset.attrs["dt"] = 0.002
    assert detect_format(v73) == "mat"
    ds = read_any(v73)
    assert ds.format == "mat" and _close(ds.data, data)


# --------------------------------------------------------------------------
# 4. RAW
# --------------------------------------------------------------------------
def test_raw():
    out = _prepare_out()
    rng = np.random.default_rng(11)
    ns, ntr = 300, 40
    t = np.arange(ns, dtype=np.float32)[:, None]
    r = np.arange(ntr, dtype=np.float32)[None, :]
    data = (np.sin(2.0 * np.pi * t / 45.0 + 0.4 * r) * 1000.0
            + 20.0 * rng.standard_normal((ns, ntr))).astype(np.float32)
    raw = out / "sample.raw"  # 注意是道优先 (ntr, ns) 的裸字节
    raw.write_bytes(np.ascontiguousarray(data.T).tobytes())

    assert detect_format(raw) == "raw"
    ds = read_any(raw)
    assert ds.format == "raw" and ds.source_kind == "raw"
    assert ds.confidence <= 0.45, ds.confidence
    assert any("按裸二进制推断" in w for w in ds.warnings), ds.warnings
    assert ds.data.ndim == 2 and ds.data.shape[0] == ns, ds.data.shape
    assert ds.data.shape[1] >= 2
    assert np.isfinite(ds.data).all()
    assert ds.dt is None and ds.dz is None

    ds2 = read_raw(raw)
    assert ds2.data.shape == ds.data.shape
    # 裸二进制没有头，截断也必须警告
    ds3 = read_raw(raw, max_traces=5)
    assert ds3.data.shape[1] == 5
    assert any("截断" in w for w in ds3.warnings)

    # 全零 RAW：不崩，候选分数低
    zeros = out / "zeros.bin"
    zeros.write_bytes(np.zeros(ns * 12, dtype="<f4").tobytes())
    try:
        dz = read_raw(zeros)
    except ValueError:
        pass  # 实在没有候选也允许报错
    else:
        assert dz.confidence <= 0.45


# --------------------------------------------------------------------------
# 5. analyze_binary
# --------------------------------------------------------------------------
def test_analyze_binary():
    out = _prepare_out()
    files = _real_files()
    if files:
        info = analyze_binary(files[0])
        assert info["format"] == "segy"
        assert info["segy"] is not None and info["segy"]["ns"] == 4000
        assert info["segy"]["n_traces_est"] == 436
        assert info["confidence"] > 0.8, info["confidence"]
        assert info["suggestions"]

    # 未知二进制：确定性随机字节，不应抛异常
    unknown = out / "mystery.xyz"
    unknown.write_bytes(bytes((np.arange(6000, dtype=np.int64) * 37 % 251).astype(np.uint8)))
    assert detect_format(unknown) == "unknown"
    info = analyze_binary(unknown)
    assert info["format"] == "unknown"
    assert 0.0 <= info["confidence"] <= 0.45, info["confidence"]
    assert info["suggestions"] and info["hexdump"]
    assert info["magic_hex"] and len(info["magic_ascii"]) == 16

    # 不存在的文件必须报 ValueError
    try:
        analyze_binary(out / "not_there.bin")
    except ValueError as exc:
        assert "不存在" in str(exc)
    else:
        raise AssertionError("analyze_binary 对不存在的文件应抛 ValueError")


# --------------------------------------------------------------------------
# 手动运行入口（pytest 未安装时）
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        print(f"\n===== {name} =====")
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)
