import torch
import torch.nn as nn


"""
Dropout 层（随机失活）

class torch.nn.Dropout(p=0.5, inplace=False)

Dropout 是一种常用的正则化方法，用于减轻神经网络的过拟合，并防止神经元之间形成过强的共同适应（co-adaptation）。它的核心思想是在训练过程中，随机暂时关闭一部分神经元。

工作方式
1. 训练模式（model.train()）
   - 对输入张量中的每个元素独立采样一个伯努利随机变量。
   - 以概率 p 将该元素置为 0；以概率 1 - p 保留该元素。
   - 为了保持输出的期望值不变，保留的元素会除以 1 - p，即缩放为
     input / (1 - p)。这种方式称为 inverted dropout。
   - 每次前向传播都会重新生成随机掩码，因此同一个输入的输出可能不同。

2. 评估模式（model.eval()）
   - 不再随机丢弃元素，也不进行额外缩放。
   - Dropout 层等价于恒等函数，直接返回输入。

参数
    p (float):元素被置零的概率，取值范围为 [0, 1)。默认值为 0.5。
    inplace (bool):是否直接在输入张量上执行操作。默认值为 False。通常保留默认值，以避免原地操作影响计算图中其他仍需使用输入的运算。

形状
    输入:  (*)，输入可以是任意形状。
    输出:  (*)，输出与输入具有相同的形状。

重要区别
    nn.Dropout 默认是逐元素随机失活：输入中的元素独立决定是否置零。
    如果需要按通道随机失活，应使用 nn.Dropout1d、nn.Dropout2d 或nn.Dropout3d；
    不要把普通 Dropout 的逐元素失活与通道级失活混为一谈。

示例
m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)

# 训练模式：约 20% 的元素被置零，输出仍保持 (20, 16) 的形状
m.train()
output_train = m(input)

# 评估模式：Dropout 关闭，输出与 input 相同
m.eval()
output_eval = m(input)
"""


# 一个最小的可运行示例
if __name__ == "__main__":
    torch.manual_seed(0)

    dropout = nn.Dropout(p=0.2)
    input = torch.randn(20, 16)

    dropout.train()
    output_train = dropout(input)

    dropout.eval()
    output_eval = dropout(input)

    print("输入形状:", input.shape)
    print("训练模式输出形状:", output_train.shape)
    print("评估模式是否与输入相同:", torch.equal(output_eval, input))
