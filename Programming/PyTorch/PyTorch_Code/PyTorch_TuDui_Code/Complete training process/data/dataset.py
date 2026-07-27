import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_SIZE,
    VAL_SIZE,
)


_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
])
TRANSFORM_EVAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
])


def build_train_val_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """创建使用相同索引、不同预处理的训练集和验证集。"""
    # 两个独立实例使训练和验证能够应用不同的 transform。
    train_full = torchvision.datasets.CIFAR10(
        DATA_DIR, train=True, transform=TRANSFORM_TRAIN, download=False
    )
    val_full = torchvision.datasets.CIFAR10(
        DATA_DIR, train=True, transform=TRANSFORM_EVAL, download=False
    )

    if TRAIN_SIZE + VAL_SIZE != len(train_full):
        raise ValueError(
            f"TRAIN_SIZE + VAL_SIZE 必须等于 {len(train_full)}，"
            f"当前为 {TRAIN_SIZE + VAL_SIZE}。"
        )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    train_indices = indices[:TRAIN_SIZE]
    val_indices = indices[TRAIN_SIZE:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_test_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """创建最终测试使用的 CIFAR-10 测试集 DataLoader。"""
    test_dataset = torchvision.datasets.CIFAR10(
        DATA_DIR, train=False, transform=TRANSFORM_EVAL, download=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader
