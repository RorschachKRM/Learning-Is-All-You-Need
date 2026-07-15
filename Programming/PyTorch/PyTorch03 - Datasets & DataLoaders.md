---
title: 数据集、数据加载器
source: https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html
tags:
  - clippings
  - Dataset
  - DataLoader
  - PyTorch
---
[入门基础](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) || [快速入门](https://docs.pytorch.ac.cn/tutorials/beginner/basics/quickstart_tutorial.html) || [张量 (Tensors)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) || **数据集与数据加载器** || [转换 (Transforms)](https://docs.pytorch.ac.cn/tutorials/beginner/basics/transforms_tutorial.html) || [构建模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) || [自动求导](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) || [优化](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) || [保存与加载模型](https://docs.pytorch.ac.cn/tutorials/beginner/basics/saveloadrun_tutorial.html)

# 数据集与数据加载器

- 处理数据样本的代码往往会变得杂乱且难以维护；理想情况下，我们希望将数据集代码与模型训练代码解耦，以获得更好的可读性和模块化。
- PyTorch 提供了两个数据原语： `torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset` ，它们允许你使用预加载的数据集以及自定义数据。
- `Dataset` 用于存储样本及其对应的标签，而 `DataLoader` 则在 `Dataset` 之上封装了一个可迭代对象，以便轻松访问样本。

PyTorch 的领域库提供了许多预加载的数据集（如 FashionMNIST），这些数据集是 `torch.utils.data.Dataset` 的子类，并实现了特定于该数据的函数。
它们可用于模型原型设计和基准测试。你可以在这里找到它们： [图像数据集](https://pytorch.ac.cn/vision/stable/datasets.html) 、 [文本数据集](https://pytorch.ac.cn/text/stable/datasets.html) 和 [音频数据集](https://pytorch.ac.cn/audio/stable/datasets.html)

# 加载数据集

以下是如何从 TorchVision 加载 [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) 数据集的示例。
Fashion-MNIST 是 Zalando 文章图像数据集，包含 60,000 个训练样本和 10,000 个测试样本。每个样本由 28×28 的灰度图像和来自 10 个类别之一的相关标签组成。

我们使用以下参数加载 [FashionMNIST 数据集](https://pytorch.ac.cn/vision/stable/datasets.html#fashion-mnist)

- `root` 是存储训练/测试数据的路径，
- `train` 指定是训练数据集还是测试数据集，
- `download=True` 表示如果数据在 `root` 中不可用，则从互联网下载数据。
- `transform` 和 `target_transform` 分别指定特征和标签的转换方式
	- 示例中使用了 PyTorch **`torchvision.transforms.v2`**（新版 API）来构建一个转换流水线。具体分两步执行：
	1. `v2.ToImage()`
		将 FashionMNIST 的原始数据（PIL Image，28×28 灰度图）转换为 `torchvision.tv_tensors.Image` 对象：
		- 自动补上通道维度，形状从 `(28, 28)` → `(1, 28, 28)`（CHW 格式）
		- 数据类型保持为 `uint8`，像素值范围 `[0, 255]`
	2. `v2.ToDtype(torch.float32, scale=True)`
		将张量转换为 `float32` 并做归一化：
		- `scale=True`：因为输入是 `uint8`，会自动除以 255，将像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`
		- 最终数据类型：`float32`

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

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
```


# 迭代与可视化数据集

我们可以像列表一样手动索引 `Datasets` ： `training_data[index]` 。我们使用 `matplotlib` 来可视化训练数据中的一些样本。
```python
# 标签映射字典。FashionMNIST的标签是数字0~9，这个字典把数字翻译成可读的文字名称。比如标签2→"Pullover"（套头衫）。
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 创建画布
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    """
    len(training_data):60000（训练集样本总数）
    torch.randint(0, 60000, size=(1,)):tensor([20417])，随机生成一个[0, 59999]`区间的整数的 1 元素张量
    .item():20417，把单元素张量提取为 Python 普通整数
    """
    
    img, label = training_data[sample_idx]
    """
    training_data是一个FashionMNIST对象，继承了Dataset。
    training_data[20417]内部调用了__getitem__(20417)，返回一个元组(图像张量, 标签)。
    自动解包：img拿到图像张量，label拿到整数标签（如5）。
	img的形状：(1, 28, 28)（单通道 28×28，已经过 ToImage 和 ToDtype 转换），值的范围[0.0, 1.0]。
    """
    
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    
    plt.imshow(img.squeeze(), cmap="gray")
    """
    img.squeeze()：去掉那个多余的通道维度（单通道只有一个值，压掉即可）。
    imshow显示 2D 灰度图需要的是(H, W)形状，而img是(C, H, W)即(1, 28, 28)，squeeze()去掉第一维变成(28, 28)，imshow 才能正确渲染。
    cmap="gray"：指定灰阶色彩映射，值0.0显示黑色，1.0显示白色。
    """
    
plt.show()
```

## image.squeeze()
去掉那个多余的通道维度（单通道只有一个值，压掉即可：把大小为 1 的维度压掉）

```
整体流程：
创建 8×8 英寸画布
    │
    ├─ 循环 9 次 ─────────────────────
    │ ① 随机抽一个样本索引 (0~59999)    
    │ ② 从 Dataset 取出 (图像, 标签)    
    │ ③ 在 3×3 网格的第 i 位放一个子图    
    │ ④ 标题 = 标签名 (如 "Sandal")     
    │ ⑤ 隐藏坐标轴                     
    │ ⑥ 显示压扁成 (28,28) 的灰度图     
    └────────────────────────────────
    │
    ▼
一次性渲染弹出窗口
```

![Bag, Ankle Boot, Sandal, Shirt, Pullover, Sneaker, Trouser, Bag, Coat](https://docs.pytorch.ac.cn/tutorials/_images/sphx_glr_data_tutorial_001.png)

---

# 为你的文件创建自定义数据集

自定义 Dataset 类必须实现三个函数： \_\_init\_\_ 、 \_\_len\_\_ 和 \_\_getitem\_\_ 。
看看这个实现：FashionMNIST 图像存储在目录 `img_dir` 中，它们的标签分别存储在 CSV 文件 `annotations_file` 中。

在接下来的部分中，我们将详细拆解这些函数中发生了什么。
```python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

## \_\_init\_\_

\_\_init\_\_ 函数在实例化 Dataset 对象时运行一次。
我们初始化包含图像的目录、注释文件以及两种转换（下一节将详细介绍）。

labels.csv 文件看起来像这样
```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

## \_\_len\_\_

\_\_len\_\_ 函数返回数据集中**样本的数量**。

示例
```python
def __len__(self):
    return len(self.img_labels)
```

## \_\_getitem\_\_

\_\_getitem\_\_ 函数加载并返回数据集中**给定索引 `idx` 处的样本**。
`Dataset`（如 `training_data`）每次只能取**一个样本**：
```python
img, label = training_data[0]   # 拿第 0 个样本
img, label = training_data[1]   # 拿第 1 个样本
```

根据索引，它确定图像在磁盘上的位置，使用 `decode_image` 将其转换为张量，从 `self.img_labels` 中的 csv 数据中检索相应的标签，对其调用转换函数（如果适用），最后以元组形式返回张量图像和对应的标签。

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

### `.iloc[idx, 0]` ——pandas 两种索引方式之一：按位置坐标
`.iloc` = **按整数位置索引**（integer location），不是按列名。
```
self.img_labels.iloc[idx, 0]
                      ↑   ↑
                     行   列
```

|代码|含义|结果示例|
|---|---|---|
|`.iloc[0, 0]`|第 0 行、第 0 列|`"tshirt1.jpg"`|
|`.iloc[0, 1]`|第 0 行、第 1 列|`0`|
|`.iloc[5, 0]`|第 5 行、第 0 列|`"tshirt6.jpg"`|
|`.iloc[5, 1]`|第 5 行、第 1 列|`0`|
```python
img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # 取文件名
label = self.img_labels.iloc[idx, 1]         # 取标签
```
- `iloc[idx, 0]` → 取出第 `idx` 行的**文件名**，拼上目录路径得到完整文件路径
- `iloc[idx, 1]` → 取出第 `idx` 行的**标签**数字

> 简单记：`iloc[行号, 列号]`，数字对数字，按位置拿。


### `.loc[idx, 0]` ——补充 pandas 两种索引方式之二：按名称/标签

如果 DataFrame 有**行名**和**列名**，就可以用名字来定位，而不是数字坐标。

给同一个 CSV 加上表头，对比一下：
```python
# 有列名的情况
#   filename         label
# 0 tshirt1.jpg      0
# 1 tshirt2.jpg      0
# 2 ankleboot999.jpg 9
```

| 方式      | 写法                      | 含义                                              |
| ------- | ----------------------- | ----------------------------------------------- |
| `.iloc` | `df.iloc[1, 0]`         | 第 1 行、第 0 列 → `"tshirt2.jpg"`                   |
| `.loc`  | `df.loc[1, "filename"]` | 行标签为 `1`、列名为 `"filename"` 的格子 → `"tshirt2.jpg"` |

`.loc` 用的是**名字**而不是坐标位置——列名叫什么就写什么，不关心它是第几列。

---

# 使用 DataLoader 为训练准备数据

`Dataset` 一次检索**一个样本的特征和标签**。
在训练模型时，我们通常希望以“小批量”（minibatches）的形式传递样本，在每个 epoch 重新打乱数据以减少模型过拟合，并使用 Python 的 `multiprocessing` 来加速数据检索。`DataLoader` 就是干这个的包装层——把 Dataset 包起来，自动帮你组批次、打乱、并行加载。

`DataLoader` 是一个**可迭代对象**，它通过简单的 API 为我们抽象了这种复杂性。
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

"""
参数：
	training_data：数据来源
	batch_size=64：每次迭代返回 64 张图 + 64 个标签
	shuffle=True：打乱，每个 epoch 开始前随机重排数据
"""
```

# 迭代 DataLoader

我们已经将该数据集加载到 `DataLoader` 中，并可以根据需要遍历数据集。
下面的每次迭代都会返回一批 `train_features` 和 `train_labels` （分别包含 `batch_size=64` 个特征和标签）。
```python
for images, labels in train_dataloader:
#    ↑           ↑
# 形状 (64, 1, 28, 28)  形状 (64,)
# 64 张图摞成一个大张量     64 个标签
```
整个数据集 60000 张 → `60000 ÷ 64 = 938` 批次（最后一批只有 48 张，PyTorch 默认不丢弃）。

因为我们指定了 `shuffle=True` ，所以在遍历完所有批次后，数据会被打乱（若要更精细地控制数据加载顺序，请查看 [采样器 (Samplers)](https://pytorch.ac.cn/docs/stable/data.html#data-loading-order-and-sampler) ）。
	**为什么要 shuffle？** 如果数据按类别排序（0 类一堆、1 类一堆…），不打乱的话模型会先学完一个类别再学下一个，梯度更新会有偏置，训练效果差。每个 epoch 打乱一次，确保每批都是随机混合的样本。

```
流程总结
Dataset (60000 个独立样本)
 │
 │  DataLoader 包装
 │
 ▼
┌─────────────────────────────────────────────┐
│  batch 1:  样本 #34821, #992, #1204, ...   │  ← 64 张
│  batch 2:  样本 #56102, #77, #44091, ...   │  ← 64 张
│  batch 3:  样本 #892, #33001, #6, ...      │  ← 64 张
│  ...
│  batch 938: 样本 #..., #...                │  ← 48 张（最后一批）
└─────────────────────────────────────────────┘
```

```python
# 显示图像和标签
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

## 代码详解：

```python
train_features, train_labels = next(iter(train_dataloader))
```
这一行包含了两个嵌套操作，从内往外看：

**第 1 步：`iter(train_dataloader)`**
`DataLoader` 是**可迭代对象**但不是迭代器本身（就像 list 不是迭代器）。`iter()` 把它转成迭代器，准备好被遍历。

|概念|举例|能直接 next() 吗？|
|---|---|---|
|可迭代对象|`[1,2,3]`, `DataLoader`|❌ 不能|
|迭代器|`iter([1,2,3])`|✅ 能|
```python
iter(train_dataloader)  # → 返回一个迭代器，内部指向第 0 批
```

**第 2 步：`next(迭代器)`**
从迭代器中取出**下一项**。因为是第一次调用，拿到的就是第 1 批数据：
```python
next(iter(train_dataloader))
# → (images, labels)  一个包含两个张量的元组
```

**第 3 步：解包**
```python
train_features, train_labels = 这个元组
# → train_features: 形状 (64, 1, 28, 28)  值范围 [0.0, 1.0]
# → train_labels:    形状 (64,)           值 0~9 的整数
```

> 为什么不用 for 循环？  
> `next(iter(...))` 是"只取一批看一眼"的模式。完整训练时会写 `for batch in train_dataloader:` 来遍历全部 938 批，但这里只是演示，取一批就够了。

```
核心逻辑：取一批 → 拿第一个 → 压缩维度 → 画出来，目的只是快速看一眼数据长什么样，确认加载正确。

DataLoader 迭代器
     │
     │  next()
     ▼
┌─────────────────────────────────────────────┐
│  train_features: (64, 1, 28, 28)            │
│  train_labels:   (64,)                      │
└─────────────────────────────────────────────┘
     │               │
     │  [0]          │  [0]
     ▼               ▼
  img: (1,28,28)   label: tensor(9)
     │
     │  squeeze()
     ▼
  img: (28, 28)  ──→  imshow  ──→  窗口显示
```

![data tutorial](https://docs.pytorch.ac.cn/tutorials/_images/sphx_glr_data_tutorial_002.png)

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 9
```

## 用for循环怎么实现以上

```python
for train_features, train_labels in train_dataloader:
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    break   # 只看第一批就停
```

Python 自动帮你做了三件事：
```
① iter(obj)       → 把可迭代对象变成迭代器
② next(迭代器)     → 一次次取下一个值
③ StopIteration   → 取完了自动结束循环，不会报错
```
所以用 `for` 写的时候，`iter` 和 `next` 一样都没少，只是**你看不到了**——Python 在底层替你干了。

## 训练时迭代dataloder代码示例

训练时的完整 `for` 循环是这样的：
```python
# 训练一个 epoch
for batch_idx, (images, labels) in enumerate(train_dataloader):
    # 1. 把数据移到 GPU
    images, labels = images.to(device), labels.to(device)
    
    # 2. 前向传播
    outputs = model(images)          # 模型预测
    loss = loss_fn(outputs, labels)  # 算损失
    
    # 3. 反向传播
    optimizer.zero_grad()   # 清空上一轮的梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数

    # 4. 每 100 批打印一次进度
    if batch_idx % 100 == 0:
        print(f"batch {batch_idx}/{len(train_dataloader)}, loss: {loss.item():.4f}")
```

### 按行解释

```python
for batch_idx, (images, labels) in enumerate(train_dataloader):
```

|部分|含义|示例值|
|---|---|---|
|`batch_idx`|第几批，来自 `enumerate`|`0, 1, 2, ..., 937`|
|`images`|这一批的图像张量|shape `(64, 1, 28, 28)`|
|`labels`|这一批的标签张量|shape `(64,)`|
循环跑完 = 一个 epoch 结束。`shuffle=True` 的话，下个 epoch 数据顺序不一样。


```python
    images, labels = images.to(device), labels.to(device)
```
`device` 是你训练用的设备：
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
把数据搬到 GPU（如果有的话），否则在 CPU 上慢几十倍。


```python
    outputs = model(images)
    loss = loss_fn(outputs, labels)
```
模型吃一批图，输出预测结果。`loss_fn`（比如 `CrossEntropyLoss`）算出预测和真实标签的差距。


```python
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
这三行是固定的标准三连：

|步骤|干什么|
|---|---|
|`zero_grad()`|把上一轮算的梯度清零（不清会累加，越攒越乱）|
|`backward()`|自动求导，计算每个参数的梯度|
|`step()`|沿梯度方向更新参数，让 loss 下降|


### 完整训练代码

把训练集和测试集都跑起来：
```python
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()  # 切换到训练模式
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_one_epoch(model, dataloader, loss_fn, device):
    model.eval()   # 切换到评估模式
    total_loss, correct = 0, 0
    with torch.no_grad():   # 测试时不计算梯度，省显存
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"Test loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

# 训练 N 个 epoch
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
    test_one_epoch(model, test_dataloader, loss_fn, device)
```
