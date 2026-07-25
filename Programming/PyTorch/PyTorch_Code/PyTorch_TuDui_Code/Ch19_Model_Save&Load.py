"""
PyTorch 模型保存与加载示例

本示例使用新版 torchvision 的 VGG16 预训练模型，演示：
1. 推荐方式：只保存和加载 model.state_dict()；
2. 保存 checkpoint：同时保存模型、优化器和训练进度；
3. 保存完整模型对象（了解即可，实际项目通常不推荐）。

注意：
- torch.save() 保存的是 PyTorch 对象；
- 只保存 state_dict 时，加载前必须先重新创建完全相同的模型结构；
- 只加载可信来源的 .pth/.pt 文件。完整模型加载可能执行文件中保存的代码。
"""

from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


NUM_CLASSES = 10
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def build_vgg16_for_cifar10(use_pretrained=True):
    """创建一个最终输出 10 类的 VGG16 模型。"""

    # 新版 torchvision：
    # weights=VGG16_Weights.DEFAULT -> 加载 ImageNet 预训练权重
    # weights=None                 -> 不加载预训练权重，随机初始化
    weights = VGG16_Weights.DEFAULT if use_pretrained else None
    model = models.vgg16(weights=weights)

    # 原始 VGG16 最后一层是 Linear(4096, 1000)，改为 CIFAR-10 的 10 类。
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        NUM_CLASSES,
    )
    return model


# -----------------------------------------------------------------------------
# 1. 创建模型
# -----------------------------------------------------------------------------

model = build_vgg16_for_cifar10(use_pretrained=True)


# -----------------------------------------------------------------------------
# 2. 推荐方式：保存和加载 state_dict
# -----------------------------------------------------------------------------

# state_dict 是一个保存了模型各层参数的字典。
# 只保存参数文件通常更小、更稳定，也更容易在不同代码中复用。
state_dict_path = CHECKPOINT_DIR / "vgg16_cifar10_state_dict.pth"
torch.save(model.state_dict(), state_dict_path)
print(f"模型参数已保存到：{state_dict_path}")


# 加载 state_dict 时，必须先重新创建相同的模型结构，再加载参数。
loaded_model = build_vgg16_for_cifar10(use_pretrained=False)
state_dict = torch.load(
    state_dict_path,
    map_location="cpu",
    weights_only=True,
)
loaded_model.load_state_dict(state_dict)

# 推理前切换到评估模式，关闭 Dropout 等训练行为。
loaded_model.eval()
print("state_dict 加载成功，模型已切换为 eval 模式。")


# -----------------------------------------------------------------------------
# 3. 保存 checkpoint：用于中断后继续训练
# -----------------------------------------------------------------------------

# 如果只保存 model.state_dict()，只能恢复模型参数，不能恢复优化器状态和训练进度。继续训练时，通常还要保存 epoch、optimizer_state_dict 和 loss。
# 保存 checkpoint是同时保存模型参数、优化器参数、当前 epoch、loss。
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

checkpoint_path = CHECKPOINT_DIR / "vgg16_cifar10_checkpoint.pth"
checkpoint = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": None,
}
torch.save(checkpoint, checkpoint_path)
print(f"训练 checkpoint 已保存到：{checkpoint_path}")


# 加载 checkpoint，恢复模型、优化器和训练轮数。
resume_model = build_vgg16_for_cifar10(use_pretrained=False)
resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=1e-4)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
resume_model.load_state_dict(checkpoint["model_state_dict"])
resume_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1

print(f"checkpoint 加载成功，可以从第 {start_epoch} 个 epoch 继续训练。")


# -----------------------------------------------------------------------------
# 4. 保存完整模型对象：了解即可
# -----------------------------------------------------------------------------

# 这种方式会把模型结构和参数一起保存，因此加载时不需要手动重新创建模型。
# 但它依赖 Python 类路径和代码环境，换机器或改代码后容易失效。
full_model_path = CHECKPOINT_DIR / "vgg16_cifar10_full_model.pth"
torch.save(model, full_model_path)
print(f"完整模型已保存到：{full_model_path}")

# 只加载可信来源的完整模型，并显式关闭 weights_only 限制。
# 在 PyTorch 新版本中，torch.load 默认更偏向只加载权重；
# 加载完整模型需要 weights_only=False。
loaded_full_model = torch.load(
    full_model_path,
    map_location="cpu",
    weights_only=False,
)
loaded_full_model.eval()
print("完整模型加载成功。")


# -----------------------------------------------------------------------------
# 5. 训练循环中常见的保存位置
# -----------------------------------------------------------------------------

# 下面是训练时的典型写法，当前只作为示例注释，不会实际执行：
#
# for epoch in range(num_epochs):
#     model.train()
#     for images, labels in train_loader:
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     torch.save(
#         {
#             "epoch": epoch,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "loss": loss.item(),
#         },
#         CHECKPOINT_DIR / f"vgg16_epoch_{epoch}.pth",
#     )
