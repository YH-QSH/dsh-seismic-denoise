"""CNNv2：分类型噪声去除。

每一类噪声单独成场并带标签，因此可以：
  ① 训练噪声类型分类器（识别"这一段里有什么噪声"）
  ② 训练分离网络（把每类噪声各自估计出来，再按需减去）

区别于一刀切的全频段去噪（CNNv1）。
"""

__version__ = "2.0.0"

from . import config, utils  # noqa: F401

__all__ = ["config", "utils", "__version__"]
