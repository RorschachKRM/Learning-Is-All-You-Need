"""MSELoss 和 CrossEntropyLoss 的基本用法示例。

MSELoss（Mean Squared Error，均方误差）
--------------------------------------
主要用于回归问题，计算预测值与真实值之间平方误差的平均值：

    MSE = 1 / n * sum((prediction - target) ** 2)

torch.nn.MSELoss(reduction='mean')

参数：
    reduction：指定输出的缩减方式，可选值为：
        - 'none'：不进行缩减，返回每个元素的损失；
        - 'mean'：返回所有元素损失的平均值（默认值）；
        - 'sum'：返回所有元素损失之和。

示例：
    prediction = torch.tensor([2.5, 3.0, 4.5])
    target = torch.tensor([3.0, 3.5, 4.0])
    loss = nn.MSELoss()(prediction, target)


CrossEntropyLoss（交叉熵损失）
------------------------------
主要用于多分类问题，用来衡量模型预测的类别分布与真实类别之间的差异。

模型输出应为 logits，即各类别的原始分数，不需要提前进行 Softmax。
CrossEntropyLoss 内部已经包含 LogSoftmax 和 NLLLoss。

torch.nn.CrossEntropyLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean',
    label_smoothing=0.0,
)

参数：
    weight：为每个类别设置权重。必须是长度为类别数 C 的 Tensor。
    ignore_index：指定需要忽略的标签值，该标签不参与损失计算和梯度计算。
    reduction：可选值为 'none'、'mean' 或 'sum'，含义与 MSELoss 相同。
    label_smoothing：标签平滑系数，范围为 [0.0, 1.0]，默认值为 0.0。

使用类别索引作为 target 时：
    - 输入形状通常为 [batch_size, num_classes]；
    - target 形状通常为 [batch_size]；target 中的值必须属于 [0, num_classes)，数据类型应为 torch.long。
    - 输出：如果 reduction 为 ‘none’，输出形状为 ()、(N) 或 (N,d1,d2,...,dK)（取决于输入形状），否则为标量。

二分类通常使用 BCEWithLogitsLoss。
"""

import torch
import torch.nn as nn


# ==================== MSELoss 示例：回归问题 ====================

# 模型预测值和真实值的形状可以是 [N, C, H, W] 等，只要两者形状一致即可。
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = inputs.reshape(1, 1, 1, 3)
targets = targets.reshape(1, 1, 1, 3)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print("MSELoss:", result_mse)


# ================= CrossEntropyLoss 示例：多分类问题 ================

# x 是两个样本对三个类别的 logits，形状为 [batch_size, num_classes]。
x = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.3, 0.2, 0.1],
], dtype=torch.float32)
print(x.shape)

# y 是真实类别的索引，0 表示第 1 类，1 表示第 2 类，依此类推。
y = torch.tensor([1, 0], dtype=torch.long)  # 此处表示：第一个样本的真实类别是 1；第二个样本的真实类别是 0

loss_cross_entropy = nn.CrossEntropyLoss()
result_cross_entropy = loss_cross_entropy(x, y)

print("CrossEntropyLoss:", result_cross_entropy)
