"""整剖面分块推理（多通道版）：输出 K+1 个通道。"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .config import K


def _起总长(total, patch, stride):
    if total <= patch:
        return [0]
    s = list(range(0, total - patch + 1, max(1, stride)))
    if s[-1] != total - patch:
        s.append(total - patch)
    return s


def _补(a, mh, mw):
    ph, pw = max(0, mh - a.shape[-2]), max(0, mw - a.shape[-1])
    if not ph and not pw:
        return a
    mode = "reflect" if (ph < a.shape[-2] and pw < a.shape[-1]) else "edge"
    try:
        return np.pad(a, ((0, 0), (0, ph), (0, pw)), mode=mode)
    except (ValueError, TypeError):
        return np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="edge")


@torch.no_grad()
def 分离剖面(model, data, patch=(256, 64), stride=None, device="cpu",
           norm="section", batch=8):
    """输入 (nt, nz) → 输出 (K+1, nt, nz)：[干净, 各噪声分量]。"""
    a = np.asarray(data, np.float32)
    nt, nz = a.shape
    s = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1.0
    inp = a / s

    pt, px = int(patch[0]), int(patch[1])
    st, sx = (pt // 2, px // 2) if stride is None else stride
    b = _补(inp[None], pt, px)[0]
    nt2, nz2 = b.shape

    ts, xs = _起总长(nt2, pt, st), _起总长(nz2, px, sx)
    wt = np.hanning(pt).astype(np.float32) + 1e-3
    wx = np.hanning(px).astype(np.float32) + 1e-3
    w2 = np.outer(wt, wx)

    acc = np.zeros((K + 1, nt2, nz2), np.float64)
    wsum = np.zeros((nt2, nz2), np.float64)
    model.eval()
    model.to(device)

    buf, coords = [], []

    def flush():
        if not buf:
            return
        xb = torch.from_numpy(np.stack(buf)[:, None]).to(device)
        ob = model(xb).detach().cpu().numpy()
        for (i, j), o in zip(coords, ob):
            for c in range(K + 1):
                acc[c, i:i + pt, j:j + px] += o[c] * w2
            wsum[i:i + pt, j:j + px] += w2
        buf.clear()
        coords.clear()

    for i in ts:
        for j in xs:
            buf.append(np.ascontiguousarray(b[i:i + pt, j:j + px]))
            coords.append((i, j))
            if len(buf) >= batch:
                flush()
    flush()

    out = acc / np.maximum(wsum, 1e-8)[None]
    return (out * s).astype(np.float32)[:, :nt, :nz]


def 按类别重建(noisy: np.ndarray, pred: np.ndarray, 权重=None) -> np.ndarray:
    """clean = noisy - Σ w_k * n_k。权重可来自分类器（按需只去某几类）。"""
    if 权重 is None:
        return pred[0]
    w = np.asarray(权重, np.float64).reshape((-1, 1, 1))
    return (noisy - np.sum(pred[1:] * w, axis=0)).astype(np.float32)


@torch.no_grad()
def 分类剖面(model, data, patch=(256, 64), stride=(128, 32), device="cpu",
          batch=16) -> np.ndarray:
    """滑窗分类，返回每类的存在概率 (K,)。"""
    a = np.asarray(data, np.float32)
    s = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1.0
    inp = a / s
    pt, px = patch
    b = _补(inp[None], pt, px)[0]
    nt2, nz2 = b.shape
    ts, xs = _起总长(nt2, pt, pt // 2), _起总长(nz2, px, px // 2)

    model.eval()
    model.to(device)
    buf, probs = [], []
    with torch.no_grad():
        for i in ts:
            for j in xs:
                buf.append(np.ascontiguousarray(b[i:i + pt, j:j + px]))
                if len(buf) >= batch:
                    p = torch.sigmoid(model(torch.from_numpy(np.stack(buf))[:, None].to(device)))
                    probs.append(p.cpu().numpy())
                    buf.clear()
        if buf:
            p = torch.sigmoid(model(torch.from_numpy(np.stack(buf))[:, None].to(device)))
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, 0).mean(0).astype(np.float32)
