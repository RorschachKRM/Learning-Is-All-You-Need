---
title: 优化器
tags:
  - DDPM
  - Optimization
  - Optimizer
---
# Adam（Adaptive Moment Estimation）
（代码 1 来源：EM_deeplearning_beifen\DDPM\train_ddpm.py)
```PYTHON
    # model
    model = ConditionalUNet(img_channels=1, base_ch=64, time_dim=256, cond_dim=64).to(
        device
    )

    diffusion = build_diffusion(timesteps, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
```

| 组件   | 说明                                                            |
| ---- | ------------------------------------------------------------- |
| 算法   | **Adam**（Adaptive Moment Estimation）                          |
| 优化对象 | `model.parameters()` — ConditionalUNet 的所有可训练参数               |
| 学习率  | `lr`（默认 1e-4）                                                 |
| 其他参数 | 使用 PyTorch 默认值：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$ |
**为什么用 Adam？** DDPM 训练本质上是一个回归任务（预测噪声），Adam 的自适应学习率特性能在不同参数尺度上稳定收敛，是扩散模型训练的标配选择。