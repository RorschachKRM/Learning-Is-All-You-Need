# CIFAR-10 图像分类训练项目

这是一个使用 PyTorch 训练卷积神经网络（`EasyNN`）完成 CIFAR-10 十分类任务的学习项目。

项目将训练、验证、最终测试、数据处理、模型定义与 checkpoint 管理解耦：训练时只用训练集更新参数，用验证集选择最佳模型；训练结束后才使用测试集做一次最终评估。

## 任务与数据集

CIFAR-10 包含 10 个类别的彩色图片，每张图片大小为 `32 × 32`、通道数为 3。原始数据划分为：

| 数据来源 | 数量 | 在本项目中的用途 |
| --- | ---: | --- |
| `train=True` | 50,000 | 按固定随机种子划为 45,000 张训练集和 5,000 张验证集 |
| `train=False` | 10,000 | 测试集，只在训练结束后评估一次 |

数据集默认应位于项目外层目录的 `CIFAR10/` 下，即：

```text
Learning-PyTorch/
├── CIFAR10/
└── Complete training process/
```

当前代码使用 `download=False`，因此需要先准备好 CIFAR-10 数据集。

## 项目结构

```text
Complete training process/
├── README.md                 # 本说明文档
├── config.py                 # 路径和超参数的集中配置
├── train.py                  # 训练入口：训练 + 验证 + 保存最佳模型
├── test.py                   # 测试入口：加载最佳模型后做最终测试
├── data/
│   ├── __init__.py           # 将 data 标记为 Python 包
│   └── dataset.py            # 数据集、数据增强、训练/验证划分和 DataLoader
├── models/
│   ├── __init__.py           # 导出 EasyNN
│   └── easy_nn.py            # EasyNN 网络结构定义
├── utils/
│   ├── __init__.py           # 将 utils 标记为 Python 包
│   ├── train_epoch.py        # 单个 epoch 的训练逻辑
│   ├── evaluate.py           # 验证集和测试集共用的评估逻辑
│   └── checkpoint.py         # checkpoint 的保存和加载
└── checkpoints/
    └── best_model.pth        # 训练时自动创建：验证集表现最好的 checkpoint
```

## 每个文件的职责

### `config.py`

集中保存不应散落在各个文件中的配置，例如：

- 数据集、日志、checkpoint 路径；
- `BATCH_SIZE = 64`；
- `TRAIN_SIZE = 45_000` 与 `VAL_SIZE = 5_000`；
- `EPOCHS = 10`、`LEARNING_RATE = 1e-2`；
- `RANDOM_SEED = 42`，用于复现训练/验证集的划分。

调整训练轮数、batch size 或学习率时，优先修改这个文件。

### `data/dataset.py`

负责创建数据集和 `DataLoader`。

- `TRANSFORM_TRAIN`：训练集使用 `RandomCrop(32, padding=4)` 和 `RandomHorizontalFlip()` 做数据增强，再执行归一化；
- `TRANSFORM_EVAL`：验证集和测试集只进行张量转换与归一化，不做随机增强，因此每次评估结果稳定；
- `build_train_val_dataloaders()`：使用一次固定随机排列，将 CIFAR-10 的 50,000 张训练数据划分为不重叠的 train/val 索引；
- `build_test_dataloader()`：创建 CIFAR-10 官方的 10,000 张测试集加载器。

训练集和验证集使用两个独立的 `CIFAR10` 实例，因此可以拥有不同的 transform，同时仍共享同一套划分索引。

### `models/easy_nn.py`

定义用于分类的 `EasyNN` 网络。它接受形状为 `[batch_size, 3, 32, 32]` 的图片，输出形状为 `[batch_size, 10]` 的分类 logits。

### `utils/train_epoch.py`

定义 `train_one_epoch()`。它在训练集上完成一次完整遍历，并执行：前向传播、计算损失、反向传播、优化器更新，以及训练损失和准确率的统计。

### `utils/evaluate.py`

定义 `evaluate()`，供验证和测试共用。它会启用 `model.eval()` 和 `torch.inference_mode()`，因此不会计算梯度或更新模型参数；返回平均 loss 与 accuracy。

### `utils/checkpoint.py`

负责保存和读取 checkpoint。一个 checkpoint 包含：

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_accuracy": best_val_accuracy,
}
```

加载时使用 `weights_only=True`，以受限反序列化方式安全读取当前这种由 tensor、数字和普通字典组成的 checkpoint。该模块也兼容此前仅保存 `model.state_dict()` 的 `.pth` 文件。

### `train.py`

训练入口，流程如下：

```text
训练集 → train_one_epoch() → 验证集 → evaluate()
                                      ↓
                            验证准确率更高？
                                      ↓
                           保存 best_model.pth
```

它使用 GPU（若可用）或 CPU，损失函数为 `CrossEntropyLoss`，优化器为 SGD，并把训练/验证指标写入 TensorBoard 日志。

### `test.py`

测试入口。它不参与模型选择，而是：

1. 加载 `checkpoints/best_model.pth`；
2. 在从未参与训练和验证的官方测试集上运行 `evaluate()`；
3. 输出最终的测试损失、测试准确率，以及该最佳模型来自哪个 epoch。

## EasyNN 模型说明

`EasyNN` 是一个三层卷积特征提取器加两层全连接分类器：

| 顺序 | 层 | 输出形状（单张图片） | 作用 |
| ---: | --- | --- | --- |
| 0 | 输入 | `3 × 32 × 32` | CIFAR-10 RGB 图片 |
| 1 | `Conv2d(3, 32, 5, padding=2)` + ReLU | `32 × 32 × 32` | 提取低层边缘、纹理等特征 |
| 2 | `MaxPool2d(2)` | `32 × 16 × 16` | 下采样，减少空间尺寸 |
| 3 | `Conv2d(32, 32, 5, padding=2)` + ReLU | `32 × 16 × 16` | 学习更复杂的局部模式 |
| 4 | `MaxPool2d(2)` | `32 × 8 × 8` | 第二次下采样 |
| 5 | `Conv2d(32, 64, 5, padding=2)` + ReLU | `64 × 8 × 8` | 提取更高层的图像特征 |
| 6 | `MaxPool2d(2)` | `64 × 4 × 4` | 第三次下采样 |
| 7 | `Flatten()` | `1024` | 展平特征，`64 × 4 × 4 = 1024` |
| 8 | `Linear(1024, 64)` + ReLU | `64` | 融合全局特征 |
| 9 | `Linear(64, 10)` | `10` | 输出 10 个类别的 logits |

最后一层没有手写 `Softmax`，这是正确的：`CrossEntropyLoss` 内部已经完成了适合训练的 `log_softmax` 与负对数似然计算。预测类别时，通过 `outputs.argmax(dim=1)` 取 logits 最大值所在的类别即可。

## 训练、验证与测试的区别

| 阶段 | 使用的数据 | 是否更新参数 | 目的 |
| --- | --- | ---: | --- |
| 训练 | 45,000 张训练集 | 是 | 通过反向传播学习模型参数 |
| 验证 | 5,000 张验证集 | 否 | 选择表现最好的 epoch/模型 |
| 测试 | 10,000 张官方测试集 | 否 | 最终、客观地报告模型效果 |

测试集不能用来挑选最佳 epoch，否则测试结果会被间接“看过”，不再是独立的最终评估。

## 如何运行

在 PowerShell 中进入项目目录后运行：

```powershell
cd "D:\VScode_Files\Learning-PyTorch\Complete training process"
& ..\learnpytorchvenv\python.exe train.py
```

训练完成后，运行最终测试：

```powershell
& ..\learnpytorchvenv\python.exe test.py
```

训练会自动创建 `checkpoints/best_model.pth`。如果尚未训练就运行 `test.py`，会提示找不到该模型文件。

可选：查看 TensorBoard 训练曲线：

```powershell
& ..\learnpytorchvenv\python.exe -m tensorboard.main --logdir ..\logs_CIFAR10
```

然后在浏览器打开命令行显示的地址（通常为 `http://localhost:6006`）。

## 依赖

项目至少需要：

```text
torch
torchvision
tensorboard
```

建议在项目已有的 `learnpytorchvenv` 环境中运行，以保证 PyTorch 与 torchvision 的版本匹配。
