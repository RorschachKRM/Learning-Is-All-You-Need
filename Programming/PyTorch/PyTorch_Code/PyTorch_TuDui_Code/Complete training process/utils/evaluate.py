import torch


@torch.inference_mode()
def evaluate(model, dataloader, loss_fn, device):
    """在验证集或测试集上评估模型，返回平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
