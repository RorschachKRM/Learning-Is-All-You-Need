import torch
from torch import nn

from config import BEST_CHECKPOINT_PATH
from data.dataset import build_test_dataloader
from models.easy_nn import EasyNN
from utils.checkpoint import load_checkpoint
from utils.evaluate import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    test_loader = build_test_dataloader()
    model = EasyNN().to(device)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint = load_checkpoint(BEST_CHECKPOINT_PATH, model, device)
    metrics = evaluate(model, test_loader, loss_fn, device)

    print("=" * 38)
    print("最终测试结果")
    print(f"Test Loss:     {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    if checkpoint["epoch"] is not None:
        print(f"最佳模型来自第 {checkpoint['epoch']} 个 epoch")
        print(f"最佳验证准确率：{checkpoint['best_val_accuracy']:.2%}")
    print("=" * 38)


if __name__ == "__main__":
    main()
