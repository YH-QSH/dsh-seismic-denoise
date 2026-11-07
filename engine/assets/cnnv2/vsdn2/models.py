"""CNNv2 模型：噪声类型分类器 + 分量分离网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import K, 模型参数


class 反射卷积(nn.Module):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__()
        self.pad = (k // 2) * d
        self.conv = nn.Conv2d(cin, cout, k, padding=0, dilation=d)

    def forward(self, x):
        if self.pad:
            x = F.pad(x, (self.pad,) * 4, mode="reflect")
        return self.conv(x)


def _bn(c, use=True):
    return nn.BatchNorm2d(c) if use else nn.Identity()


class 双卷积(nn.Module):
    def __init__(self, cin, cout, use_bn=True):
        super().__init__()
        self.f = nn.Sequential(
            反射卷积(cin, cout), _bn(cout, use_bn), nn.ReLU(True),
            反射卷积(cout, cout), _bn(cout, use_bn), nn.ReLU(True),
        )

    def forward(self, x):
        return self.f(x)


# ---------------------------------------------------------------- 分类器
class 噪声分类器(nn.Module):
    """patch -> K 个 logit（多标签：同一段可以同时有面波和工频）。"""

    def __init__(self, ch=32, n_layers=6, use_bn=True, k=K):
        super().__init__()
        layers = [反射卷积(1, ch), nn.ReLU(True)]
        d = 1
        for i in range(n_layers):
            d = [1, 2, 4][i % 3]
            layers += [反射卷积(ch, ch, d=d), _bn(ch, use_bn), nn.ReLU(True)]
            if i in (2, 4):
                layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch), nn.ReLU(True), nn.Linear(ch, k),
        )

    def forward(self, x):
        return self.head(self.body(x))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 分离器
class 分离网络(nn.Module):
    """U-Net，输出 K+1 通道：[干净波场, 各噪声分量...]。"""

    def __init__(self, base=32, depth=3, use_bn=True, k=K):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.inc = 双卷积(1, chs[0], use_bn)
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), 双卷积(chs[i], chs[i + 1], use_bn))
             for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [双卷积(chs[i + 1] + chs[i], chs[i], use_bn) for i in range(depth - 1, -1, -1)]
        )
        self.outc = nn.Conv2d(chs[0], k + 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(skips[-1]))
        h = skips[-1]
        for i, up in enumerate(self.ups):
            s = skips[len(skips) - 2 - i]
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([s, h], dim=1))
        return self.outc(h)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def 建模型(mp: 模型参数, 类型: str = "分离"):
    if 类型 == "分类":
        return 噪声分类器(ch=mp.分类器_通道, n_layers=mp.分类器_层数, use_bn=mp.bn)
    return 分离网络(base=mp.分离器_base, depth=mp.分离器_depth, use_bn=mp.bn)
