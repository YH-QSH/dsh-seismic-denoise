"""CNNv2 配置：分类型噪声去除。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from .utils import load_json, save_json

REF_DT = 0.002
REF_DZ = 5.0

# 噪声类别（顺序 = 网络输出通道顺序）
类别 = ["面波", "随机噪声", "工频干扰", "散射绕射", "折射波"]
类别_英 = ["groundroll", "random", "mono50", "scatter", "refraction"]
K = len(类别)


@dataclass
class 合成参数:
    # 网格：训练固定在"规范网格"上（2 ms / 5 m）。
    # 任何别的采样参数，推理时先重采样到规范网格再送网络，算完再插值回去。
    # 这样视倾角只由速度决定，与采集参数解耦——同一套权重能吃各种 dt/dz。
    nt: int = 512
    nz: int = 96
    dt: float = REF_DT
    dz_range: Tuple[float, float] = (REF_DZ, REF_DZ)
    地面概率: float = 0.35         # 地面炮集
    面源VSP概率: float = 0.40      # 面源 VSP（复刻 MATLAB 里 shuju*.m 那一族）

    # 面源 VSP 专属
    面源_源道距: float = 26.0                                    # 炮点到相邻道的水平距离（m）
    面源_v1_range: Tuple[float, float] = (1600.0, 2600.0)        # 首层速度
    面源_层深_range: Tuple[float, float] = (400.0, 2800.0)
    面源_起伏_amp: Tuple[float, float] = (40.0, 160.0)           # 曲面层正弦起伏（m）
    面源_衰减: Tuple[float, float] = (0.35, 0.95)                # exp(-decay*t)
    面源_面波v: Tuple[float, float] = (600.0, 900.0)
    面源_面波f: Tuple[float, float] = (4.0, 8.0)
    面源_面波amp: Tuple[float, float] = (0.3, 0.9)

    # 介质
    vp_range: Tuple[float, float] = (2200.0, 5500.0)
    f0_range: Tuple[float, float] = (18.0, 45.0)
    decay_range: Tuple[float, float] = (0.05, 0.6)
    spreading_z0: float = 800.0

    # 有效波
    n_layers_range: Tuple[int, int] = (2, 5)
    layer_amp_range: Tuple[float, float] = (0.35, 0.9)

    # 0 面波（VSP 里即管波：都是低速强相干）
    面波_v_range: Tuple[float, float] = (250.0, 1400.0)
    面波_amp_range: Tuple[float, float] = (0.4, 2.2)
    面波_f_lo: float = 4.0
    面波_f_ratio: Tuple[float, float] = (0.25, 0.75)   # 相对 f0
    面波_存在概率: float = 0.9

    # 1 随机噪声
    随机_白比: Tuple[float, float] = (0.55, 0.9)
    带限截止: Tuple[float, float] = (0.06, 0.4)

    # 2 工频
    工频_f_range: Tuple[float, float] = (45.0, 60.0)
    工频_amp_range: Tuple[float, float] = (0.15, 1.0)
    工频_存在概率: float = 0.55

    # 3 散射 / 绕射
    散射数_range: Tuple[int, int] = (0, 160)
    散射_amp_range: Tuple[float, float] = (0.06, 0.35)
    散射_孔径: Tuple[int, int] = (3, 15)

    # 4 折射波（仅地面几何有意义）
    折射_存在概率: float = 0.5
    折射_amp_range: Tuple[float, float] = (0.3, 1.0)
    折射_v_range: Tuple[float, float] = (1200.0, 2600.0)

    # 总体强度
    snr_range: Tuple[float, float] = (-10.0, 40.0)
    相干占比: Tuple[float, float] = (0.2, 0.9)   # 相干噪声占总噪声能量比例

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class 数据参数:
    patch_t: int = 256
    patch_x: int = 64
    每剖面patch数: int = 8
    norm: str = "section"
    seed_train: int = 1000
    seed_val: int = 2000
    seed_test: int = 3000
    测试集条数: int = 60


@dataclass
class 模型参数:
    分类器_通道: int = 32
    分类器_层数: int = 6
    分离器_base: int = 32
    分离器_depth: int = 3
    bn: bool = True


@dataclass
class 训练参数:
    设备: str = "auto"
    batch: int = 48
    lr: float = 1e-3
    weight_decay: float = 1e-5
    workers: int = 12
    amp: bool = True
    分类器_轮数: int = 8
    分类器_每轮步数: int = 200
    分离器_轮数: int = 26
    分离器_每轮步数: int = 250
    grad_clip: float = 1.0
    预热步数: int = 100

    合成: 合成参数 = field(default_factory=合成参数)
    数据: 数据参数 = field(default_factory=数据参数)
    模型: 模型参数 = field(default_factory=模型参数)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "训练参数":
        seq = lambda x: tuple(x) if isinstance(x, list) else x  # noqa: E731
        syn = 合成参数(**{k: seq(v) for k, v in d.get("合成", {}).items()})
        dat = 数据参数(**{k: seq(v) for k, v in d.get("数据", {}).items()})
        mod = 模型参数(**d.get("模型", {}))
        top = {k: v for k, v in d.items() if k not in ("合成", "数据", "模型")}
        return cls(合成=syn, 数据=dat, 模型=mod, **top)

    @classmethod
    def load(cls, path: str | Path) -> "训练参数":
        return cls.from_dict(load_json(path))


def 预设(name: str = "default") -> 训练参数:
    cfg = 训练参数()
    if name == "quick":
        cfg.合成.nt = 320
        cfg.合成.nz = 64
        cfg.分类器_轮数 = 1
        cfg.分类器_每轮步数 = 20
        cfg.分离器_轮数 = 1
        cfg.分离器_每轮步数 = 20
        cfg.batch = 8
        cfg.workers = 0
    return cfg
