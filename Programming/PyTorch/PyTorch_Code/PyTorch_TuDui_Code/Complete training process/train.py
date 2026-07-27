import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import (
    BEST_CHECKPOINT_PATH,
    EPOCHS,
    LEARNING_RATE,
    LOG_DIR,
    LOG_INTERVAL,
    RANDOM_SEED,
)
from data.dataset import build_train_val_dataloaders
from models.easy_nn import EasyNN
from utils.checkpoint import save_checkpoint
from utils.evaluate import evaluate
from utils.train_epoch import train_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_train_val_dataloaders()
    print(f"训练集数量：{len(train_loader.dataset)}")
    print(f"验证集数量：{len(val_loader.dataset)}")

    model = EasyNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(str(LOG_DIR))

    best_val_accuracy = -1.0
    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(EPOCHS):
            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                start_step=global_step,
                writer=writer,
                log_interval=LOG_INTERVAL,
            )
            val_metrics = evaluate(model, val_loader, loss_fn, device)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"train loss: {train_metrics['loss']:.4f}, "
                f"train acc: {train_metrics['accuracy']:.2%} | "
                f"val loss: {val_metrics['loss']:.4f}, "
                f"val acc: {val_metrics['accuracy']:.2%}"
            )
            writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch + 1)
            writer.add_scalar("train/epoch_accuracy", train_metrics["accuracy"], epoch + 1)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch + 1)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch + 1)

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    BEST_CHECKPOINT_PATH,
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    best_val_accuracy=best_val_accuracy,
                )
                print(f"已保存最佳模型：{BEST_CHECKPOINT_PATH}")
    finally:
        writer.close()

    print(f"训练完成，用时：{time.time() - start_time:.1f}s")
    print(f"最佳验证准确率：{best_val_accuracy:.2%}")


if __name__ == "__main__":
    main()

