from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader
from model import EasyNN
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
核心规则:
模型 (parameters)  ─────┐
输入数据 (imgs)    ─────┤── 全在 GPU 或全在 CPU
标签   (targets)   ─────┘
"""

# ========== 设备配置 ==========
# 1. 检测可用设备，GPU 优先，没有则回退 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：{}".format(device))
if device.type == "cuda":
    print("  GPU 型号：{}".format(torch.cuda.get_device_name(0)))
    print("  显存大小：{:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# 准备数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
train_dataset = torchvision.datasets.CIFAR10(ROOT / "CIFAR10", train=True, transform=transform_train, download=False)
test_dataset = torchvision.datasets.CIFAR10(ROOT / "CIFAR10", train=False, transform=transform_test, download=False)

# 数据集数量
train_dataset_length = len(train_dataset)
test_dataset_length = len(test_dataset)
print("训练数据集数量：{}".format(train_dataset_length))
print("测试数据集数量：{}".format(test_dataset_length))

# 准备dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 2. 模型实例化后，立刻搬到设备上
MyNN = EasyNN().to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(MyNN.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 10

writer = SummaryWriter(str(ROOT / "logs_CIFAR10"))

for i in range(epoch):
    print("--------第 {} epoch训练开始------".format(i+1))

    MyNN.train()
    for data in train_dataloader:
        imgs, targets = data
        # 3. 训练时，输入和标签都要搬到设备上
        """
        .to() 不会修改原数据，imgs.to(device) 返回的是新 tensor，不会原地修改，所以必须赋值回去
        """
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = MyNN(imgs)
        train_loss = loss_fn(outputs, targets)

        # 优化模型
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 ==0:
            print("当前训练次数:{}, Loss:{}".format(total_train_step, train_loss.item()))
            writer.add_scalar("train_Loss", train_loss.item(), total_train_step)


    MyNN.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 4. 测试时同样要搬到设备上
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = MyNN(imgs)
            test_loss = loss_fn(outputs, targets)
            total_test_loss += test_loss.item() * imgs.size(0) # 所有样本的 loss 之和

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    avg_test_loss = total_test_loss / test_dataset_length
    accuracy_rate = total_accuracy / test_dataset_length

    print("测试集上Loss：{}".format(avg_test_loss))
    print("测试集上Accuracy：{}".format(accuracy_rate))
    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy_rate, total_test_step)
    total_test_step = total_test_step + 1


    state_dict_path = CHECKPOINT_DIR / "MyNN_{}.pth".format(i)
    torch.save(MyNN.state_dict(), state_dict_path)
    print("模型已保存")

writer.close()
