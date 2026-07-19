---
title: 优化模型参数
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html
tags:
  - clippings
  - PyTorch
  - Optimization
---
[基础入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) || [变换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动微分 (Autograd)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || **优化 (Optimization)** || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

# 优化模型参数

既然我们已经有了模型和数据，现在是时候通过在数据上优化模型参数来训练、验证和测试我们的模型了。
训练模型是一个迭代的过程：在每一次迭代中，模型会对输出进行猜测，计算猜测的误差（ *损失* ），收集误差相对于其参数的导数（如我们在 [上一节](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) 所见），并使用梯度下降法 **优化** 这些参数。
有关此过程的更详细讲解，请查看这个关于 [3Blue1Brown 的反向传播](https://www.youtube.com/watch?v=tIeHLnjs5U8) 视频。

 

# 先决条件代码

我们加载了来自 [数据集与数据加载器](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) 和 [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) 章节的代码。
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```


# 超参数

超参数是可调整的参数，让您可以控制模型优化过程。不同的超参数值会影响模型训练和收敛速度（ [阅读更多](https://pytorch.ac.cn/tutorials/beginner/hyperparameter_tuning_tutorial.html) 关于超参数调优的内容）。

我们为训练定义了以下**超参数**：
- **迭代次数 (Epochs)** - 在整个数据集上进行迭代的次数。
- **批次大小 (Batch Size)** - 在参数更新前，通过网络传播的数据样本数量。
- **学习率 (Learning Rate)** - 在每个批次/迭代中更新模型参数的幅度。较小的值会导致学习速度缓慢，而较大的值可能会在训练过程中导致不可预测的行为。
```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```


# 优化循环

设置好超参数后，我们就可以通过优化循环来训练和优化模型。
优化循环的每一次迭代称为一个 **epoch（轮次）** 。

每个轮次包含两个主要部分：
- **训练循环** - 遍历训练数据集，并尝试收敛到最优参数。
- **验证/测试循环** - 遍历测试数据集，检查模型性能是否在提高。

让我们先简要熟悉一下训练循环中用到的一些概念。直接跳转查看优化循环的。

## 损失函数

当面对训练数据时，未经训练的网络很可能无法给出正确的答案。 **损失函数 (Loss function)** 用于衡量所得结果与目标值之间的差异程度，而我们训练的目标就是最小化损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实的标签值进行比较。

常见的损失函数包括回归任务中的 [nn.MSELoss](https://pytorch.ac.cn/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) （均方误差）和分类任务中的 [nn.NLLLoss](https://pytorch.ac.cn/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) （负对数似然）。 [nn.CrossEntropyLoss](https://pytorch.ac.cn/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 结合了 `nn.LogSoftmax` 和 `nn.NLLLoss` 。

我们将模型的**输出逻辑值 (logits)** 传递给 `nn.CrossEntropyLoss` ，它将归一化这些 logits 并计算预测误差。
```python
# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
```

## 优化器

优化是调整模型参数以减少每一步训练中模型误差的过程。
**优化算法** 定义了这一过程如何执行（在本例中，我们使用随机梯度下降法）。
所有优化逻辑都封装在 `optimizer` 对象中。在此，我们使用 SGD 优化器；此外，PyTorch 中还有许多 [不同的优化器](https://pytorch.ac.cn/docs/stable/optim.html) ，例如 ADAM 和 RMSProp，它们对不同类型的模型和数据效果更好。

我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

## 优化过程三步
在训练循环内，优化过程分三步进行：
- 调用 `optimizer.zero_grad()` 重置模型参数的梯度。默认情况下，梯度会累加；为了防止重复计算，我们在每次迭代时显式地将它们归零。
- 通过调用 `loss.backward()` 对预测损失进行反向传播。PyTorch 会存储损失相对于每个参数的梯度。
- 一旦我们有了梯度，就调用 `optimizer.step()` ，根据反向传播中收集的梯度来调整参数。



# 完整实现

我们定义了循环执行优化代码的 `train_loop` ，以及评估模型在测试数据上表现的 `test_loop` 。
```python
"""
train：前向 → 算 loss → 反向传播 → 更新参数 → 每 100 batch 打印当前 loss
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 将模型设置为训练模式——这对于批量归一化（训练时用当前 batch 的均值/方差，评估时用训练期间累积的全局均值/方差）和dropout层（训练时随机丢弃神经元，评估时全部保留）至关重要
    # 在这种情况下并非必要，但为了最佳实践，还是加上了
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

		# 日志输出，每隔 100 个 batch 打印一次训练进度
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

"""前向 → 统计 loss 和正确数 → 准确率 + 平均 loss"""
def test_loop(dataloader, model, loss_fn):
    # 将模型设置为评估模式——这对于批量归一化和dropout层至关重要
    # 在这种情况下并非必要，但为了最佳实践，还是加上了
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 使用 torch.no_grad() 评估模型可确保在测试模式下不计算梯度
    # 此外，对于 requires_grad=True 的张量，还可以减少不必要的梯度计算和内存使用
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # 累加每个batch的loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累加预测正确的样本

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
`(pred.argmax(1) == y)` 这一行的流程：

|步骤|操作|示例|
|---|---|---|
|`pred`|模型输出，shape `(batch_size, 10)`|每行是 10 个类别的 logits|
|`.argmax(1)`|沿 dim=1 取最大值索引，即**预测的类别**|`[3, 7, 1, ...]`|
|`== y`|与真实标签比较|`[True, False, True, ...]`|
|`.type(torch.float)`|转成浮点 1.0 / 0.0|`[1.0, 0.0, 1.0, ...]`|
|`.sum().item()`|求和 = 本 batch 正确个数|`47`|


我们初始化损失函数和优化器，并将其传递给 `train_loop` 和 `test_loop` 。您可以随意增加轮次 (epochs) 的数量来跟踪模型性能的提升。
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 2.298335  [   64/60000]
loss: 2.289470  [ 6464/60000]
loss: 2.272484  [12864/60000]
loss: 2.273880  [19264/60000]
loss: 2.247583  [25664/60000]
loss: 2.227010  [32064/60000]
loss: 2.233613  [38464/60000]
loss: 2.197765  [44864/60000]
loss: 2.198511  [51264/60000]
loss: 2.178389  [57664/60000]
Test Error:
 Accuracy: 46.3%, Avg loss: 2.163991

Epoch 2
-------------------------------
loss: 2.166300  [   64/60000]
loss: 2.164202  [ 6464/60000]
loss: 2.106268  [12864/60000]
loss: 2.128797  [19264/60000]
loss: 2.077504  [25664/60000]
loss: 2.022356  [32064/60000]
loss: 2.045981  [38464/60000]
loss: 1.968023  [44864/60000]
loss: 1.977939  [51264/60000]
loss: 1.916258  [57664/60000]
Test Error:
 Accuracy: 57.1%, Avg loss: 1.907327

Epoch 3
-------------------------------
loss: 1.931413  [   64/60000]
loss: 1.915910  [ 6464/60000]
loss: 1.793170  [12864/60000]
loss: 1.838027  [19264/60000]
loss: 1.739939  [25664/60000]
loss: 1.684238  [32064/60000]
loss: 1.700629  [38464/60000]
loss: 1.598467  [44864/60000]
loss: 1.626210  [51264/60000]
loss: 1.529740  [57664/60000]
Test Error:
 Accuracy: 61.2%, Avg loss: 1.540385

Epoch 4
-------------------------------
loss: 1.596202  [   64/60000]
loss: 1.574571  [ 6464/60000]
loss: 1.413586  [12864/60000]
loss: 1.493455  [19264/60000]
loss: 1.384402  [25664/60000]
loss: 1.367305  [32064/60000]
loss: 1.377125  [38464/60000]
loss: 1.299336  [44864/60000]
loss: 1.336691  [51264/60000]
loss: 1.244146  [57664/60000]
Test Error:
 Accuracy: 63.4%, Avg loss: 1.265495

Epoch 5
-------------------------------
loss: 1.332922  [   64/60000]
loss: 1.325499  [ 6464/60000]
loss: 1.150386  [12864/60000]
loss: 1.262408  [19264/60000]
loss: 1.147163  [25664/60000]
loss: 1.163012  [32064/60000]
loss: 1.176008  [38464/60000]
loss: 1.116805  [44864/60000]
loss: 1.156523  [51264/60000]
loss: 1.076140  [57664/60000]
Test Error:
 Accuracy: 64.5%, Avg loss: 1.093752

Epoch 6
-------------------------------
loss: 1.157416  [   64/60000]
loss: 1.167994  [ 6464/60000]
loss: 0.978276  [12864/60000]
loss: 1.116740  [19264/60000]
loss: 1.000959  [25664/60000]
loss: 1.026564  [32064/60000]
loss: 1.050967  [38464/60000]
loss: 1.000912  [44864/60000]
loss: 1.040937  [51264/60000]
loss: 0.969616  [57664/60000]
Test Error:
 Accuracy: 65.8%, Avg loss: 0.983067

Epoch 7
-------------------------------
loss: 1.035882  [   64/60000]
loss: 1.066518  [ 6464/60000]
loss: 0.861274  [12864/60000]
loss: 1.020486  [19264/60000]
loss: 0.909911  [25664/60000]
loss: 0.931637  [32064/60000]
loss: 0.969104  [38464/60000]
loss: 0.926418  [44864/60000]
loss: 0.962395  [51264/60000]
loss: 0.897819  [57664/60000]
Test Error:
 Accuracy: 67.2%, Avg loss: 0.908232

Epoch 8
-------------------------------
loss: 0.946642  [   64/60000]
loss: 0.996810  [ 6464/60000]
loss: 0.778834  [12864/60000]
loss: 0.954176  [19264/60000]
loss: 0.849679  [25664/60000]
loss: 0.863269  [32064/60000]
loss: 0.911837  [38464/60000]
loss: 0.876940  [44864/60000]
loss: 0.906460  [51264/60000]
loss: 0.846428  [57664/60000]
Test Error:
 Accuracy: 68.3%, Avg loss: 0.854870

Epoch 9
-------------------------------
loss: 0.878244  [   64/60000]
loss: 0.945053  [ 6464/60000]
loss: 0.717747  [12864/60000]
loss: 0.906241  [19264/60000]
loss: 0.806933  [25664/60000]
loss: 0.812294  [32064/60000]
loss: 0.868711  [38464/60000]
loss: 0.842430  [44864/60000]
loss: 0.864620  [51264/60000]
loss: 0.807422  [57664/60000]
Test Error:
 Accuracy: 69.7%, Avg loss: 0.814673

Epoch 10
-------------------------------
loss: 0.823512  [   64/60000]
loss: 0.903935  [ 6464/60000]
loss: 0.670369  [12864/60000]
loss: 0.870073  [19264/60000]
loss: 0.774233  [25664/60000]
loss: 0.772918  [32064/60000]
loss: 0.834202  [38464/60000]
loss: 0.816819  [44864/60000]
loss: 0.831830  [51264/60000]
loss: 0.776483  [57664/60000]
Test Error:
 Accuracy: 71.0%, Avg loss: 0.782795

Done!
```