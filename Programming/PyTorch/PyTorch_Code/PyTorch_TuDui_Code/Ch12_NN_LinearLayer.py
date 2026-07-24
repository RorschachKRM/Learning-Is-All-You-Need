from torch.nn import Linear

# 线性层(全连接层)
"""
可用于分类、回归

class torch.nn.Linear(
    in_features,
    out_features,
    bias=True,
    device=None,
    dtype=None,
)

功能:对输入数据的最后一个维度执行线性变换：y = xA^T + b

参数:
    in_features (int):每个输入样本的特征数量。

    out_features (int):每个输出样本的特征数量。

    bias (bool):是否学习可加性偏置。默认为 True。设置为 False 时，该层不包含偏置参数。

输入形状:(*, H_in)其中：* 表示任意数量的前置维度，也可以没有前置维度；H_in = in_features。
输出形状:(*, H_out)其中：除最后一个维度外，其余维度与输入保持一致；H_out = out_features。

示例:
    nn.Linear(128, 64)
    输入形状              输出形状
    (128,)         ->     (64,)
    (32, 128)      ->     (32, 64)
    (8, 32, 128)   ->     (8, 32, 64)

    nn.Linear(in_features, out_features) 只改变输入的最后一个维度。
"""

