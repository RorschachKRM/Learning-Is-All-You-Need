"""
使用新版 torchvision 加载 VGG16 预训练模型

本示例演示：
1. 使用 ImageNet 预训练权重创建 VGG16；
2. 使用随机权重创建 VGG16；
3. 将 VGG16 原来的 1000 分类改成 CIFAR-10 的 10 分类；
4. 使用预训练权重自带的数据预处理方式加载 CIFAR-10。

注意：本文件只演示模型和数据集的准备，没有执行训练。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import VGG16_Weights


# CIFAR-10 一共有 10 个类别
NUM_CLASSES = 10


# -----------------------------------------------------------------------------
# 1. 创建 VGG16 模型
# -----------------------------------------------------------------------------

# 新版 torchvision 的写法：
#   weights=None                         -> 不加载预训练权重，随机初始化
#   weights=VGG16_Weights.DEFAULT        -> 加载官方默认预训练权重
vgg16_random = models.vgg16(weights=None)

weights = VGG16_Weights.DEFAULT
vgg16_pretrained = models.vgg16(weights=weights)


# -----------------------------------------------------------------------------
# 2. 查看预训练权重提供的标准预处理
# -----------------------------------------------------------------------------

# 预训练模型是在 ImageNet 的图像预处理方式下训练的。
# weights.transforms() 会返回与该权重匹配的 Resize、CenterCrop、ToTensor和 Normalize 等操作，适合直接用于迁移学习。
preprocess = weights.transforms()


# -----------------------------------------------------------------------------
# 3. 加载 CIFAR-10 数据集
# -----------------------------------------------------------------------------

train_data = datasets.CIFAR10(
    root="./CIFAR10",
    train=True,
    transform=preprocess,
    download=True,
)

test_data = datasets.CIFAR10(
    root="./CIFAR10",
    train=False,
    transform=preprocess,
    download=True,
)

# DataLoader 不是本示例重点，但这样可以直接用于后续训练。
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# -----------------------------------------------------------------------------
# 4. 修改 VGG16 的最后一个分类层
# -----------------------------------------------------------------------------

# VGG16 原来的最后一层是：Linear(4096, 1000)，用于 ImageNet 的 1000 类分类。
# CIFAR-10 只有 10 类，因此直接替换最后一层。
#
# 不建议使用旧代码中的 add_module("add_linear", Linear(1000, 10)：那会保留原来的 1000 类输出层，再额外增加一个 1000 -> 10 的层。
"""
.classifier[6]：取出分类器的第 6 层，也就是最后一层Linear(in_features=4096, out_features=1000)
.in_features：读取线性层的输入特征数量
"""
random_in_features = vgg16_random.classifier[6].in_features
pretrained_in_features = vgg16_pretrained.classifier[6].in_features

"""
1. 替换随机模型的最后一层，把原来的Linear(4096, 1000)替换成Linear(4096, 10)。这个新建的 Linear(4096, 10) 会使用随机值初始化。

2. 替换预训练模型的最后一层，替换最后一层后，VGG16 的卷积层和前面分类层仍然保留预训练权重，只有新替换的最后一层是随机初始化的。
"""
vgg16_random.classifier[6] = nn.Linear(random_in_features, NUM_CLASSES)
vgg16_pretrained.classifier[6] = nn.Linear(pretrained_in_features, NUM_CLASSES)


# -----------------------------------------------------------------------------
# 5. 输出结果，确认模型已经改为 10 分类
# -----------------------------------------------------------------------------

print("预训练权重名称：", weights)
print("预训练模型的类别数量：", len(weights.meta["categories"]))
print("CIFAR-10 训练集大小：", len(train_data))
print("CIFAR-10 测试集大小：", len(test_data))
print("修改后的随机初始化模型最后一层：", vgg16_random.classifier[6])
print("修改后的预训练模型最后一层：", vgg16_pretrained.classifier[6])


# -----------------------------------------------------------------------------
# 6. 可选：检查一个 batch 的形状
# -----------------------------------------------------------------------------

# weights.transforms() 会把 CIFAR-10 的 32x32 图片调整为 VGG16 常用的 224x224 输入尺寸，因此这里的图像形状应为 [32, 3, 224, 224]。
images, labels = next(iter(train_loader))
print("一个 batch 的图像形状：", images.shape)
print("一个 batch 的标签形状：", labels.shape)


# 后续训练时可以使用：
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(vgg16_pretrained.parameters(), lr=1e-4)
# 
# outputs = vgg16_pretrained(images)
# loss = criterion(outputs, labels)
# 
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
