import torch.nn as nn
from torch.nn import BatchNorm2d
import torch

# Normalization Layers归一化层

"""
BatchNorm:数据按通道归一化，并为每个通道学习一个缩放参数和一个偏移参数。

class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, *, bias=True)

参数：
    num_features (int) – 预期输入大小 (N,C,H,W) 中的 C。

    eps (float) – 为了数值稳定性添加到分母中的值。默认值：1e-5

    momentum (float | None) – 用于 running_mean 和 running_var 计算的值。可以设置为 None 以进行累积移动平均（即简单平均）。默认为：0.1

    affine (bool) – 一个布尔值，如果设置为 True，则该模块具有可学习的仿射参数。默认值：True

    track_running_stats (bool) – 一个布尔值。当设为 True 时，该模块会跟踪运行均值和方差；当设为 False 时，该模块不跟踪此类统计量，并将统计量缓冲区 running_mean 和 running_var 初始化为 None。当这些缓冲区为 None 时，该模块在训练和评估模式下始终使用批统计量。默认值：True

    bias (bool) – 如果设置为 False，该层将不会学习加性偏置（仅在 affine 为 True 时相关）。默认值: True
形状
    输入： (N,C,H,W)
    输出： (N,C,H,W) （与输入形状相同）
"""
# 有可学习参数：γ（缩放）、β（偏移）
BN = BatchNorm2d(100)
input = torch.randn(20, 100, 35, 45)
output = BN(input)
output.shape


"""
GroupNorm
"""


"""
LayerNorm
"""