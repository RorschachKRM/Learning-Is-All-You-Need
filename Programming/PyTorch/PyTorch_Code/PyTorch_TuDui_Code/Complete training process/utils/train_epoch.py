import torch


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    start_step=0,
    writer=None,
    log_interval=100,
):
    """训练一个 epoch，返回指标和更新后的全局 step。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    global_step = start_step

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        global_step += 1

        if writer is not None and global_step % log_interval == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
    return metrics, global_step
