from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch, best_val_accuracy):
    """保存模型参数、优化器状态和训练进度，支持后续断点续训。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        },
        path,
    )


def load_checkpoint(path, model, device, optimizer=None):
    """加载 checkpoint，并可选恢复优化器状态。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件：{path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # 兼容此前只保存 state_dict 的 .pth 文件。
    if "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
        return {"epoch": None, "best_val_accuracy": None}

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
