---
title: 损失函数
tags:
  - Loss
  - MSE
  - Cross-Entropy
related:
  - "[[DDPM - General Framework#7.4 损失函数变体]]"
---
# MSE

## 1. 定义与原理

**均方误差（Mean Squared Error）**：预测值与真实值之差的平方的平均。

$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(\hat{y}_i - y_i\big)^2$$

**三个核心要点，即为全部原理：**

1. **为什么是平方**：统计学上，假设"真实值 = 预测值 + 高斯噪声"，对高斯分布做极大似然估计，取负对数后剩下的正是平方项——**最小化 MSE ⇔ 高斯噪声假设下的极大似然估计**。这与交叉熵对应伯努利/多项分布是平行关系（见 [[Loss Function#1.4 KL 散度：为什么最小化交叉熵有意义|交叉熵 1.4]]），所以选损失函数本质是选"你认为输出服从什么分布"：连续值 + 高斯 → MSE；离散类别 → 交叉熵。
2. **梯度性质**：$\frac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$，梯度与误差成正比——错得越多修正越狠，接近答案时步子自动变小，训练平滑稳定。
3. **平方的代价——对离群点敏感**：误差 10 的样本贡献是误差 1 的 **100 倍**，少数异常样本会绑架整个训练。数据脏时考虑 L1 或 Huber（见下方对比表）。

> [!warning] 何时不用 MSE
> 分类任务不要用 MSE——配合 Sigmoid/Softmax 会梯度消失且非凸，见 [[Loss Function#2.2 优美的梯度（为什么分类不用 MSE）|交叉熵 2.2]]。**MSE 管连续值，交叉熵管类别。**

## 2. 具体应用

| 应用 | 预测什么 | 说明 |
|---|---|---|
| **数值回归** | 房价、温度、年龄 | 最经典用法 |
| **图像重建** | 逐像素的像素值 | 自编码器、超分辨率、去噪 |
| **DDPM 噪声预测** ⭐ | 加到图像上的高斯噪声 $\epsilon$ | 本笔记的主角，见 2.2 |
| **强化学习** | 状态价值 / Q 值 | DQN 的 TD 误差 |
| **知识蒸馏** | 教师模型的中间特征 | 特征对齐 |

### 2.1 基础用法：回归

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(),
    nn.Linear(64, 1)           # ← 输出层无激活函数，直接输出任意实数
)
criterion = nn.MSELoss()       # reduction='mean'（默认，对所有元素求平均）

x = torch.randn(32, 10)
y = torch.randn(32, 1)         # ← 标签是 float 连续值，形状与输出一致

loss = criterion(model(x), y)
loss.backward()

# 推理：输出直接就是预测值，无需任何后处理
pred = model(x)
```

与交叉熵实战对照：**输出层不加激活、标签是 float 连续值、推理无后处理**——比分类简单得多，没有 logits/概率之分。

### 2.2 DDPM 中的 MSE ⭐

DDPM 训练目标：给图像加上已知噪声 $\epsilon$，让 U-Net 从加噪图像中**把这个噪声猜出来**：

$$\mathcal{L}_{simple} = \mathbb{E}_{t,\, x_0,\, \epsilon}\Big[\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2\Big]$$

```python
import torch.nn.functional as F

def training_step(model, x0, T, alphas_cumprod):
    t = torch.randint(0, T, (x0.shape[0],), device=x0.device)  # 随机时间步
    noise = torch.randn_like(x0)                               # 真实噪声 ε（标签！）
    x_t = q_sample(x0, t, noise)                               # 前向加噪
    pred_noise = model(x_t, t)                                 # U-Net 预测噪声
    return F.mse_loss(pred_noise, noise)                       # ← 就这一行 MSE
```

为什么 DDPM 用 MSE 而不是交叉熵？

1. 噪声 $\epsilon$ 是**连续张量**（形状同图像），不是类别
2. 反向过程被建模为**高斯分布**，从 ELBO 的 KL 散度化简下来天然就是平方项——正好对应上面第 1 点"MSE ⇔ 高斯假设下的极大似然"

详见 [[DDPM - General Framework#7.4 损失函数变体]]、[[DDPM - Training]]。

## 3. 变体对比：MSE vs L1 vs Huber

| 损失 | 公式 | 对离群点 | PyTorch | 选用场景 |
|---|---|---|---|---|
| **MSE (L2)** | $(\hat{y}-y)^2$ | 敏感 ❌ | `nn.MSELoss` | 数据干净（默认首选） |
| **MAE (L1)** | $\|\hat{y}-y\|$ | 稳健 ✅ | `nn.L1Loss` | 离群点多；梯度恒定不平滑 |
| **Huber** | 小误差用 L2，大误差用 L1 | 稳健 ✅ | `nn.SmoothL1Loss` / `nn.HuberLoss` | 两者折中（DQN、目标检测常用） |

> [!summary] 一句话总结
> **预测连续值 → MSE；输出层不加激活，标签给 float；数据有离群点换 Huber；DDPM 的损失就是对噪声的一行 `F.mse_loss`。**




# 交叉熵

> [!info] 相关笔记
> 首次遇到于 [[Programming/PyTorch/PyTorch06 - Automatic Differentiation with torch.autograd|PyTorch06 - 自动微分]] 中的 `binary_cross_entropy_with_logits`。

> [!info] 跳过原理直接看应用
> 见[[Loss Function#5. PyTorch API 对照]]、[[Loss Function#6. 实战指南：怎么选、怎么写 ⭐]]

## 1. 信息论基础：从熵到交叉熵

### 1.1 自信息（Self-Information）

一个事件发生的概率越小，它发生时携带的"信息量"越大：

$$I(x) = -\log p(x)$$

- 必然事件（$p=1$）：信息量为 0
- 罕见事件（$p \to 0$）：信息量趋于无穷大

### 1.2 熵（Entropy）

熵是**自信息的期望**，衡量一个分布的不确定性：

$$H(p) = -\sum_{x} p(x) \log p(x)$$

分布越均匀（越"不确定"），熵越大；分布越集中，熵越小。

### 1.3 交叉熵（Cross-Entropy）

用**预测分布 $q$** 去编码**真实分布 $p$** 所需的平均信息量：

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

### 1.4 KL 散度：为什么最小化交叉熵有意义

KL 散度衡量两个分布的"距离"：

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \underbrace{H(p, q)}_{\text{交叉熵}} - \underbrace{H(p)}_{\text{常数}}$$

**关键结论**：真实分布 $p$ 固定时，$H(p)$ 是常数，所以：

$$\min H(p,q) \iff \min D_{KL}(p \| q)$$

==即最小化交叉熵 = 让预测分布尽可能接近真实分布。当 $q = p$ 时交叉熵取最小值。==

> [!tip] 与 DDPM 的联系
> DDPM 的损失推导也是从最小化 KL 散度出发（变分下界 ELBO），最终简化为对噪声的 [[Loss Function#MSE|MSE]]。交叉熵和 DDPM 损失同根同源——都源于"让模型分布逼近数据分布"。


## 2. 二元交叉熵（BCE, Binary Cross-Entropy）

二分类时，真实标签 $y \in \{0, 1\}$，模型预测正类（y=1）概率 $\hat{y} = \sigma(z) \in (0,1)$，分布只有两个取值。
此处解析：
	1. 模型的原始输出z （logit）是任意实数，本身不是概率。经过 Sigmoid 压缩后变成$\hat{y} = \sigma(z) \in (0,1)$。例如$\hat{y} = 0.9$意思是：模型认为这个样本有 90% 的把握是正类.
	2. “分布”是伯努利分布：定义在{0,1}两个取值上的最简单的离散分布，一个参数就完全确定它。这里其实有**两个**伯努利分布在对比：

① **预测分布 q**（模型给出的）：

|       | x=1的概率    | x=0概率       | 特点       |
| ----- | --------- | ----------- | -------- |
| 预测分布q | $\hat{y}$ | 1-$\hat{y}$ | 又不确定性    |
| 真实分布p | y（非0即1）   | 1-y         | 退化，无不确定性 |

② **真实分布 p**（标签给出的，是一个退化分布，也叫 one-hot / 狄拉克分布）：

**情形一：这个样本的真实标签是 1（正类）**

| 取值x |    概率p(x)     |
| :-: | :-----------: |
| x=1 | 1← 100% 确定是正类 |
| x=0 |       0       |

**情形二：这个样本的真实标签是 0（负类）**

| 取值x |    概率p(x)     |
| :-: | :-----------: |
| x=1 |       0       |
| x=0 | 1← 100% 确定是负类 |
上面两个情形可以**用标签y本身来统一表达**：
$${p(x = 1)} = y，{p(x = 0)} = 1 - y$$


### **==BCE公式==**
（由通用交叉熵公式 ，求和遍历0,1两项）为：

$$\mathcal{L}_{BCE} = -\big[\, y \log \hat{y} + (1-y) \log(1-\hat{y}) \,\big]$$

**分情况理解**：

| 真实标签 | 损失 | 行为 |
|---------|------|------|
| $y=1$ | $-\log \hat{y}$ | $\hat{y} \to 1$ 损失 $\to 0$；$\hat{y} \to 0$ 损失 $\to \infty$ |
| $y=0$ | $-\log(1-\hat{y})$ | $\hat{y} \to 0$ 损失 $\to 0$；$\hat{y} \to 1$ 损失 $\to \infty$ |

直觉：**预测错得越自信，惩罚越重**（对数惩罚是无界的）。
二分类时随机变量只能取{0,1}两个值（伯努利分布），把真实标签分布 p 和模型预测分布 q 代入通用交叉熵公式，求和只剩两项，就得到了 BCE。

### 2.1 Sigmoid 函数

把任意实数 logit $z$ 压缩为概率：

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

### 2.2 优美的梯度（为什么分类不用 MSE）

BCE 对 logit $z$ 求导，Sigmoid 的导数恰好被约掉：

$$\frac{\partial \mathcal{L}_{BCE}}{\partial z} = \sigma(z) - y = \hat{y} - y$$

- 梯度 = **预测误差**，误差越大梯度越大，学得越快 ✅
- 若用 MSE + Sigmoid：$\frac{\partial \mathcal{L}}{\partial z} = (\hat{y}-y)\,\sigma'(z)$，当 $z$ 很大/很小时 $\sigma'(z) \to 0$，即使预测完全错误梯度也趋于 0（**梯度消失**，学不动）❌
- 另外 MSE + Sigmoid 是非凸的，而 BCE + Sigmoid 关于 $z$ 是凸的

## 3. 多分类交叉熵（Categorical Cross-Entropy)

$C$ 个类别，真实标签 one-hot 编码，配合 Softmax：

$$\hat{y}_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}, \qquad \mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log \hat{y}_i = -\log \hat{y}_{c}$$

由于 one-hot 中只有真实类别 $c$ 处为 1，损失就是**真实类别预测概率的负对数**（负对数似然 NLL）。

梯度同样优美：$\dfrac{\partial \mathcal{L}_{CE}}{\partial z_i} = \hat{y}_i - y_i$

## 4. 数值稳定性：为什么用 with_logits 版本

$$\text{朴素写法：} \log(\sigma(z)) \xrightarrow{z \ll 0} \log(0) = -\infty \; \Rightarrow \; \text{NaN}$$

`with_logits` 版本在内部合并 Sigmoid 和 log，使用 **log-sum-exp 技巧**：

$$\mathcal{L} = \max(z, 0) - z \cdot y + \log\big(1 + e^{-|z|}\big)$$

指数项永远是 $e^{-|z|} \le 1$，不会上溢或下溢。

## 5. PyTorch API 对照

| API | 输入 | 内含激活 | 场景 |
|-----|------|---------|------|
| `F.binary_cross_entropy` | 概率 $\in (0,1)$ | 无（需手动 Sigmoid） | 二分类 ⚠️ 数值不稳 |
| `F.binary_cross_entropy_with_logits` | logits | Sigmoid ✅ | 二分类 / 多标签（推荐） |
| `F.cross_entropy` | logits | LogSoftmax ✅ | 多分类（推荐） |
| `F.nll_loss` | log 概率 | 无（需手动 LogSoftmax） | 多分类 |

```python
import torch
import torch.nn.functional as F

z = torch.randn(3)          # logits（未经激活的原始输出）
y = torch.zeros(3)          # 标签，3 个独立的二分类任务

# ✅ 推荐：直接喂 logits，内部做 Sigmoid，数值稳定
loss = F.binary_cross_entropy_with_logits(z, y)

# ❌ 等价但数值不稳定
loss_bad = F.binary_cross_entropy(torch.sigmoid(z), y)
```



## 6. 实战指南：怎么选、怎么写 ⭐

- **二分类**：垃圾邮件识别、疾病诊断 → BCE
- **多标签分类**：一张图同时含猫和狗（各标签独立二分类）→ BCE
- **多分类**：MNIST 手写数字（类别互斥）→ Categorical CE
- **回归 / 连续值**（如 DDPM 预测噪声）→ [[Loss Function#MSE|MSE]]，不用交叉熵

### 6.1 决策表（最重要）

| 你的任务 | 判断标准 | 用什么 | 模型输出 | 标签格式 |
|---|---|---|---|---|
| **二分类** | 每个样本只回答"是/否" | `nn.BCEWithLogitsLoss` | 1 个数（logit） | `float`，0. 或 1. |
| **多标签分类** | 每个样本可同时属于多个类 | `nn.BCEWithLogitsLoss` | C 个数 | `float`，multi-hot 如 `[1,0,1]` |
| **多分类** | 每个样本只属于 C 类中的 1 类 | `nn.CrossEntropyLoss` | C 个数 | `long`，类别索引如 `2` |

> [!important] 两条铁律
> 1. 模型最后一层**不要加** Sigmoid/Softmax，直接输出裸 logits 喂给损失函数（它们内部自带）
> 2. `BCEWithLogitsLoss` 的标签要 `float`；`CrossEntropyLoss` 的标签要 `long`（整数索引，**不是 one-hot**）

### 6.2 场景一：二分类（垃圾邮件、是否患病）

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 1)          # ← 输出 1 个 logit，不加 Sigmoid！
)
criterion = nn.BCEWithLogitsLoss()

x = torch.randn(32, 20)                    # batch=32, 特征=20
y = torch.randint(0, 2, (32, 1)).float()   # ← 必须 float，形状和输出一致 (32,1)

# 训练
logits = model(x)                # (32, 1)
loss = criterion(logits, y)
loss.backward()

# 推理：此时才手动加 Sigmoid
prob = torch.sigmoid(logits)     # 概率
pred = (prob > 0.5).long()       # 0 或 1
```

### 6.3 场景二：多标签分类（一张图同时含猫、狗、车）

和二分类**完全一样的写法**，只是输出 C 个 logit，每个标签独立判断：

```python
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))  # 3 个标签
criterion = nn.BCEWithLogitsLoss()

x = torch.randn(32, 20)
y = torch.tensor([[1., 0., 1.]] * 32)   # ← multi-hot：有猫、没狗、有车

loss = criterion(model(x), y)

# 推理：每个标签各自过 0.5 门槛，可以同时多个为 1
pred = (torch.sigmoid(model(x)) > 0.5).long()   # 如 [1, 0, 1]
```

### 6.4 场景三：多分类（MNIST 十选一，类别互斥）⭐ 最常用

```python
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))  # 10 类
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))      # ← 直接给类别编号，long 型，形状 (32,)
                                     #    不是 one-hot！[0,0,0,1,0,...] ❌

logits = model(x)                    # (32, 10)
loss = criterion(logits, y)          # 内部自动做 Softmax + 取负对数
loss.backward()

# 推理：选 logit 最大的类，连 Softmax 都可以省
pred = logits.argmax(dim=1)          # (32,) 每个样本一个类别编号
```

### 6.5 三个实用参数

```python
# ① 二分类类别不平衡（如 95% 负样本）：给正类加权
nn.BCEWithLogitsLoss(pos_weight=torch.tensor([19.0]))   # 负/正 = 95/5 = 19

# ② 多分类不平衡：按类别加权
nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 5.0]))  # 稀有类权重大

# ③ 标签平滑：防过拟合，让模型别太自信
nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 6.6 高频报错对照

| 报错/症状 | 原因 | 解法 |
|---|---|---|
| `expected scalar type Long` | CrossEntropyLoss 标签给了 float | `y = y.long()` |
| `result type Float can't be cast` | BCEWithLogitsLoss 标签给了 int | `y = y.float()` |
| loss 不降、训练极慢 | 最后一层手动加了 Softmax/Sigmoid | 删掉，输出裸 logits |
| 形状报错 (32,1) vs (32,) | BCE 要求输出和标签形状完全一致 | `y.unsqueeze(1)` 或 `logits.squeeze(1)` |

> [!summary] 一句话总结
> **互斥选一个 → `CrossEntropyLoss` + 整数标签；独立判断是否 → `BCEWithLogitsLoss` + float 标签；输出永远是裸 logits。**
