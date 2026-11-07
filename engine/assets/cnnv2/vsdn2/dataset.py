"""数据集：按需实时合成（不占磁盘），以及少量固定测试集。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import synth
from .config import K, 合成参数


def _尺度(a: np.ndarray) -> float:
    s = float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))
    return s if s > 1e-12 else 1.0


class 动态数据集(Dataset):
    """每次取样本时实时合成一条剖面再裁 patch。

    好处：数据量不受磁盘限制，想扩多大就多大。
    可复现：随机数由 (seed, epoch, idx) 派生。
    """

    def __init__(self, cfg: 合成参数, patch_t=256, patch_x=64,
                 pps=8, n=4000, seed=1000, norm="section", augment=True,
                 返回标签=False):
        self.cfg = cfg
        self.pt, self.px = int(patch_t), int(patch_x)
        self.pps = int(pps)
        self.n = int(n)
        self.seed = int(seed)
        self.norm = norm
        self.augment = augment
        self.返回标签 = 返回标签
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return self.n * self.pps

    def __getitem__(self, idx):
        sec = int(idx) // self.pps
        rng = np.random.default_rng([self.seed, self.epoch, sec])
        out = synth.生成(self.cfg, int(rng.integers(0, 2**31 - 1)))

        noisy = out["noisy"]
        clean = out["clean"]
        parts = out["parts"]
        s = _尺度(noisy)

        nt, nz = noisy.shape
        i0 = int(rng.integers(0, nt - self.pt + 1)) if nt > self.pt else 0
        j0 = int(rng.integers(0, nz - self.px + 1)) if nz > self.px else 0
        sl = (slice(i0, i0 + self.pt), slice(j0, j0 + self.px))

        x = noisy[sl]
        y = np.concatenate([clean[sl][None], parts[(slice(None),) + sl]], axis=0)
        if self.augment and rng.random() < 0.5:
            x = x[:, ::-1]
            y = y[:, ::-1]

        x = (x / s).astype(np.float32)
        y = (y / s).astype(np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(x))[None]
        if self.返回标签:
            return xt, torch.from_numpy(np.asarray(out["label"], np.float32))
        return xt, torch.from_numpy(np.ascontiguousarray(y))


def 造加载器(ds, batch, shuffle, workers=0, drop_last=True):
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=max(0, workers),
              drop_last=drop_last, pin_memory=False)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


def 生成测试集(cfg: 合成参数, n: int, seed: int) -> Dict[str, np.ndarray]:
    """固定测试集（落盘，便于反复对比）。"""
    arrs = {"noisy": [], "clean": [], "label": [], "snr": [], "geo": [], "dz": []}
    for i in range(n):
        out = synth.生成(cfg, seed + i)
        arrs["noisy"].append(out["noisy"])
        arrs["clean"].append(out["clean"])
        arrs["label"].append(out["label"])
        arrs["snr"].append(out["snr_in_db"])
        arrs["geo"].append(out["geo"])
        arrs["dz"].append(out["dz"])
    return {
        "noisy": np.stack(arrs["noisy"]).astype(np.float32),
        "clean": np.stack(arrs["clean"]).astype(np.float32),
        "label": np.stack(arrs["label"]).astype(np.float32),
        "snr": np.array(arrs["snr"], np.float32),
        "geo": np.array(arrs["geo"]),
        "dz": np.array(arrs["dz"], np.float32),
    }


def 存npz(path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kw)


def 读npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}
