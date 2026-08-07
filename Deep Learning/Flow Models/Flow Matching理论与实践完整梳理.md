# Flow Matching 理论与实践完整梳理

> 目标：从训练和生成两个模块，完整梳理 Flow Matching 的理论推导逻辑、实际代码操作，以及二者为什么看起来不同却能够严格对应。

本文覆盖 Flow Matching 的核心理论—实践闭环，足以作为理解基础 Flow Matching、Conditional Flow Matching 和 Rectified Flow 的主线讲义。Optimal Transport Flow Matching、Stochastic Interpolant、条件引导和大规模网络结构属于后续专题，不影响本文主线。

---

## 一、先解决最容易造成混乱的问题

理解 Flow Matching 时，必须区分三件事：

1. **理论中用公式定义一个对象；**
2. **实际代码显式计算这个对象；**
3. **实际代码通过随机采样或神经网络间接估计这个对象。**

它们不是一回事。

> 理论里写出一个积分，不等于代码中必须把这个积分解析计算出来。很多理论积分负责说明“目标是什么、为什么正确”，实际训练则用 Monte Carlo 采样和神经网络回归绕过显式计算。

本文统一采用如下方向：

$$
t=0:\text{噪声},\qquad t=1:\text{数据}.
$$

有些论文或代码使用相反方向，此时插值方向和速度符号也会相反。

---

## 二、符号表

| 符号 | 含义 |
|---|---|
| $X_0$ | 初始噪声样本 |
| $X_1$ | 真实数据样本 |
| $p_0$ | 初始噪声分布，通常为标准高斯 |
| $p_1=p_{\mathrm{data}}$ | 目标数据分布 |
| $Z$ | 条件变量，本文主要令 $Z=(X_0,X_1)$ |
| $q(z)$ 或 $\pi(x_0,x_1)$ | 端点对的联合采样分布，也叫耦合 |
| $I_t(X_0,X_1)$ | 人为选择的随机插值 |
| $X_t$ | 时刻 $t$ 的中间状态 |
| $p_t(x\mid z)$ | 给定条件 $z$ 后的条件概率分布 |
| $p_t(x)$ | 将条件变量边缘化后的总体分布 |
| $u_t(x\mid z)$ 或 $U_t$ | 条件向量场或采样得到的条件速度 |
| $v_t(x)$ | 边缘向量场，生成时真正需要的速度场 |
| $v_\theta(x,t)$ | 神经网络对边缘向量场的近似 |

---

# 第一部分：训练模块

## 三、训练模块的最终目标

我们拥有：

- 容易采样的噪声分布 $p_0$；
- 难以计算密度、但可以从数据集采样的数据分布 $p_1=p_{\mathrm{data}}$。

希望找到一个随时间变化的向量场：

$$
v_t(x),
$$

使 ODE

$$
\boxed{
\frac{\mathrm dX_t}{\mathrm dt}=v_t(X_t)
}
$$

能够把噪声分布运输成数据分布：

$$
X_0\sim p_0
\quad\Longrightarrow\quad
X_1\sim p_{\mathrm{data}}.
$$

因此 Flow Matching 最终真正需要学习的是：

$$
\boxed{\text{边缘向量场 }v_t(x)}.
$$

实际使用神经网络 $v_\theta(x,t)$ 近似它。

---

## 四、先区分三个层次

### 4.1 单条条件轨迹

先抽取一对端点：

$$
Z=(X_0,X_1).
$$

给定具体端点：

$$
z=(x_0,x_1),
$$

人为定义一条连接它们的轨迹：

$$
X_t=I_t(x_0,x_1).
$$

例如 Rectified Flow：

$$
X_t=(1-t)x_0+tx_1.
$$

固定 $(x_0,x_1)$ 后，这是一条具体粒子轨迹。

### 4.2 所有条件轨迹形成边缘概率路径

训练时不断抽取不同端点对：

$$
(X_0,X_1)\sim q.
$$

每对端点都产生一条条件轨迹。将所有轨迹上的粒子混合起来，时刻 $t$ 的总体分布就是：

$$
p_t(x).
$$

当 $t$ 连续变化时，得到边缘概率路径：

$$
p_0\longrightarrow p_t\longrightarrow p_1.
$$

### 4.3 神经网络学习总体概率质量的速度

每条条件轨迹都有自己的速度，但无条件生成时，网络不知道真实目标端点 $X_1$，只能根据当前位置和时间输出：

$$
v_\theta(x,t).
$$

因此网络最终学习的不是某一条指定条件轨迹的速度，而是当前位置处所有可能条件速度的局部平均，即边缘向量场。

---

## 五、训练理论：完整推导逻辑

### 5.1 定义端点联合分布 $q(z)$

令：

$$
Z=(X_0,X_1).
$$

需要规定怎样抽取端点对：

$$
Z\sim q(z).
$$

如果独立采样：

$$
X_0\sim p_0,
\qquad
X_1\sim p_{\mathrm{data}},
\qquad
X_0\perp X_1,
$$

那么：

$$
\boxed{
q(x_0,x_1)
=
p_0(x_0)p_{\mathrm{data}}(x_1)
}
$$

这叫独立耦合。

更一般地，可以指定其他耦合：

$$
(X_0,X_1)\sim\pi(x_0,x_1),
$$

例如 Minibatch OT 配对、配对数据或 Reflow 产生的端点对。

$q$ 并不是训练之外额外隐藏的神秘分布，它就是“端点怎样被采样和配对”的数学表达。

### 5.2 人为设计条件路径

选择插值：

$$
X_t=I_t(X_0,X_1).
$$

一般要求：

$$
I_0(X_0,X_1)=X_0,
$$

$$
I_1(X_0,X_1)=X_1.
$$

一种通用形式为：

$$
X_t=a(t)X_0+b(t)X_1,
$$

其中：

$$
a(0)=1,\quad b(0)=0,
$$

$$
a(1)=0,\quad b(1)=1.
$$

Rectified Flow 选择最简单的直线插值：

$$
\boxed{
X_t=(1-t)X_0+tX_1
}.
$$

条件路径虽然是人为选择的，但需要满足：

- 端点分布正确；
- 中间状态容易采样；
- 条件速度容易计算；
- 路径足够平滑；
- 对应 ODE 具有较好的数值性质。

### 5.3 从插值得到条件速度

对随机插值关于时间求导：

$$
\boxed{
U_t
=
\frac{\mathrm dX_t}{\mathrm dt}
=
\partial_t I_t(X_0,X_1)
}.
$$

对一般线性组合：

$$
X_t=a(t)X_0+b(t)X_1,
$$

有：

$$
U_t=\dot a(t)X_0+\dot b(t)X_1.
$$

Rectified Flow 中：

$$
X_t=(1-t)X_0+tX_1,
$$

所以：

$$
\boxed{
U_t=X_1-X_0
}.
$$

$U_t$ 是某一对端点对应的条件速度，也是实际训练中容易计算的监督标签。

### 5.4 连续性方程的作用

条件概率路径满足：

$$
\partial_t p_t(x\mid z)
+
\nabla\cdot
\left[
p_t(x\mid z)u_t(x\mid z)
\right]
=0.
$$

它表达概率质量守恒。

工程上通常不会在每次训练时求解这个偏微分方程。更常见的逻辑是：

1. 先定义可微随机插值 $I_t$；
2. 直接求导得到条件速度；
3. 使用连续性方程从理论上说明该速度推动了对应的条件概率路径。

因此：

> 对插值求导是实际计算条件速度的常用方法；连续性方程主要用于理论刻画和证明，不是普通训练循环中需要数值求解的方程。

如果只给定概率密度路径而没有粒子插值，也可以尝试通过连续性方程寻找兼容向量场，但高维情况下求解困难，而且解一般不唯一。

### 5.5 条件分布边缘化

给定条件 $Z=z$ 后，时刻 $t$ 的条件分布为：

$$
p_t(x\mid z).
$$

将条件变量积分掉：

$$
\boxed{
p_t(x)
=
\int p_t(x\mid z)q(z)\,\mathrm dz
}
$$

得到边缘概率分布。

这个积分在理论上说明：

> 时刻 $t$ 的总体分布，是所有条件分布按照端点采样频率混合后的结果。

它不表示实际代码必须遍历整个数据集和所有噪声，显式计算出 $p_t(x)$。

### 5.6 从概率流推导边缘向量场

每个条件在位置 $x$ 处产生的概率流为：

$$
J_t(x\mid z)
=
p_t(x\mid z)u_t(x\mid z).
$$

混合所有条件后，总概率流为：

$$
J_t(x)
=
\int
p_t(x\mid z)u_t(x\mid z)q(z)
\,\mathrm dz.
$$

边缘速度 $v_t(x)$ 应满足：

$$
J_t(x)=p_t(x)v_t(x).
$$

所以：

$$
v_t(x)
=
\frac{
\int p_t(x\mid z)u_t(x\mid z)q(z)\,\mathrm dz
}{p_t(x)}.
$$

根据贝叶斯公式：

$$
p_t(z\mid x)
=
\frac{p_t(x\mid z)q(z)}{p_t(x)},
$$

得到：

$$
\boxed{
v_t(x)
=
\int u_t(x\mid z)p_t(z\mid x)\,\mathrm dz
=
\mathbb E[u_t(x\mid Z)\mid X_t=x]
}.
$$

因此，边缘向量场是条件速度按照当前位置处的后验概率进行的局部加权平均。

### 5.7 理想 Flow Matching Loss

如果能够直接得到边缘向量场 $v_t(x)$，理想损失可以写为：

$$
\boxed{
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{t,X_t\sim p_t}
\left[
\left\|v_\theta(X_t,t)-v_t(X_t)\right\|^2
\right]
}.
$$

它直接使用真实边缘速度监督网络。

但真实高维问题中通常很难显式计算：

- 数据密度 $p_{\mathrm{data}}(x)$；
- 边缘密度 $p_t(x)$；
- 后验分布 $p_t(z\mid x)$；
- 边缘向量场 $v_t(x)$。

这正是理论和实践开始分开的地方。

---

## 六、训练实践：代码实际怎么操作？

### 6.1 实际训练使用 Conditional Flow Matching

虽然边缘速度 $v_t(x)$ 很难显式计算，但条件速度 $U_t$ 通常容易计算。因此实际训练使用：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}
=
\mathbb E
\left[
\left\|v_\theta(X_t,t)-U_t\right\|^2
\right]
}.
$$

Rectified Flow 中直接使用：

$$
U_t=X_1-X_0.
$$

### 6.2 为什么用条件速度能学到边缘速度？

网络只观察：

$$
(X_t,t),
$$

并不知道产生监督标签的完整端点 $Z=(X_0,X_1)$。

均方误差下的理想最优回归函数是条件期望：

$$
v^*(x,t)
=
\mathbb E[U_t\mid X_t=x,t].
$$

而理论已经证明：

$$
\mathbb E[U_t\mid X_t=x,t]
=
v_t(x).
$$

所以：

$$
\boxed{
v^*(x,t)=v_t(x)
}.
$$

训练时虽然提供的是单条条件轨迹的速度，但网络最终回归的是边缘速度。

### 6.3 CFM Loss 与理想 FM Loss 的关系

平方误差可以分解为：

$$
\begin{aligned}
\mathcal L_{\mathrm{CFM}}
={}&
\mathbb E
\left[
\|v_\theta(X_t,t)-v_t(X_t)\|^2
\right]\\
&+
\mathbb E
\left[
\|U_t-v_t(X_t)\|^2
\right].
\end{aligned}
$$

第二项与模型参数 $\theta$ 无关，因此：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}
=
\mathcal L_{\mathrm{FM}}
+
\text{与模型参数无关的项}
}.
$$

所以二者具有相同的最优解和模型参数梯度。

Conditional Flow Matching 的核心价值是：

> 使用容易计算的条件速度，替代难以显式计算的边缘速度作为监督目标。

### 6.4 实际训练不显式计算哪些对象？

真实训练通常不会显式计算：

$$
q(z)\text{ 的密度值},qquad
p_t(x),qquad
p_t(z\mid x),qquad
v_t(x).
$$

它们主要负责定义和证明。

真实代码需要执行的是：

- 从 $q(z)$ 对应的采样过程获得端点；
- 构造中间状态 $X_t$；
- 计算条件速度 $U_t$；
- 使用 MSE 更新神经网络。

### 6.5 “从 $q(z)$ 采样”是什么意思？

基础 Rectified Flow 中：

```python
x1 = next(dataloader)
x0 = torch.randn_like(x1)
```

这两行就定义并实现了：

$$
Z=(X_0,X_1)\sim q.
$$

如果独立采样，理论上有：

$$
q(x_0,x_1)
=
p_0(x_0)p_{\mathrm{data}}(x_1).
$$

但训练不需要执行：

```python
q_value = q(x0, x1)
```

因为期望可以用 Monte Carlo 样本平均近似：

$$
\boxed{
\mathbb E_{Z\sim q}[f(Z)]
\approx
\frac1B\sum_{i=1}^{B}f(z_i),
\qquad z_i\sim q
}.
$$

这就是“只需要采样，不需要计算概率密度”的含义。

### 6.6 为什么不知道 $p_{\mathrm{data}}$ 密度也能采样？

我们不知道真实数据分布的解析密度，但拥有有限数据集：

$$
\{x_1^{(1)},\ldots,x_1^{(N)}\}.
$$

DataLoader 实际从经验分布采样：

$$
\hat p_{\mathrm{data}}(x)
=
\frac1N
\sum_{i=1}^{N}
\delta(x-x_1^{(i)}).
$$

代码：

```python
x1 = next(dataloader)
```

相当于从经验数据分布中采样。

所以我们虽然不能计算 $p_{\mathrm{data}}(x)$ 在任意图像上的密度值，却能够获得近似来自该分布的训练样本。对基于样本期望的训练已经足够。

### 6.7 Rectified Flow 的完整训练步骤

```python
import torch


def append_dims(t, x):
    return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))


def rectified_flow_loss(model, x1):
    batch_size = x1.shape[0]
    device = x1.device

    # 1. 采样噪声端点 X₀
    x0 = torch.randn_like(x1)

    # 2. 采样时间 t
    t = torch.rand(batch_size, device=device)
    t_view = append_dims(t, x1)

    # 3. 构造中间状态 Xₜ
    xt = (1.0 - t_view) * x0 + t_view * x1

    # 4. 计算条件目标速度 Uₜ
    target_velocity = x1 - x0

    # 5. 网络预测边缘速度
    predicted_velocity = model(xt, t)

    # 6. CFM / Rectified Flow Loss
    loss = torch.mean(
        (predicted_velocity - target_velocity) ** 2
    )

    return loss
```

外层优化：

```python
model.train()

for x1 in dataloader:
    x1 = x1.to(device)

    optimizer.zero_grad(set_to_none=True)

    loss = rectified_flow_loss(model, x1)

    loss.backward()
    optimizer.step()
```

### 6.8 训练理论与实践对照

| 对象 | 理论作用 | 实际是否显式计算 |
|---|---|---:|
| $p_0$ | 定义初始分布 | 不计算密度，只调用噪声采样器 |
| $p_{\mathrm{data}}$ | 定义目标分布 | 不计算密度，从 DataLoader 采样 |
| $q(z)$ | 定义端点联合分布 | 不计算密度，通过采样和配对实现 |
| $p_t(x\mid z)$ | 定义条件概率路径 | 通常不计算密度，直接构造 $X_t$ |
| $u_t(x\mid z)$ 或 $U_t$ | 条件速度 | 需要计算，作为训练标签 |
| $p_t(x)$ | 定义边缘概率路径 | 不显式计算 |
| $p_t(z\mid x)$ | 定义局部后验权重 | 不显式计算 |
| $v_t(x)$ | 理想边缘向量场 | 不显式计算，由网络学习 |
| $v_\theta(x,t)$ | 对边缘向量场的近似 | 网络实际输出 |

---

## 七、为什么理论要定义实践不计算的对象？

理论需要回答三个问题。

### 7.1 网络想学到的究竟是什么？

答案是边缘向量场：

$$
v_t(x).
$$

如果不定义它，就无法说明训练完成后网络表示什么。

### 7.2 为什么使用条件速度训练是正确的？

因为：

$$
\mathbb E[U_t\mid X_t=x,t]
=
v_t(x).
$$

理论中的后验、边缘化和概率流推导证明了这一点。

### 7.3 学到边缘向量场后为什么能生成目标分布？

因为边缘向量场满足连续性方程：

$$
\partial_t p_t(x)
+
\nabla\cdot
\left[p_t(x)v_t(x)\right]
=0.
$$

它保证 ODE 推动的分布按照目标概率路径变化。

所以这些理论对象虽然不在训练代码中显式出现，却负责证明训练方法和生成方法能够正确连接。

---

# 第二部分：生成模块

## 八、生成理论：为什么求解 ODE 能生成数据？

训练完成后，假设网络准确学习了：

$$
v_\theta(x,t)\approx v_t(x).
$$

边缘向量场满足：

$$
\partial_t p_t(x)
+
\nabla\cdot
\left[p_t(x)v_t(x)\right]
=0.
$$

考虑 ODE：

$$
\frac{\mathrm dX_t}{\mathrm dt}
=
v_t(X_t).
$$

如果初始状态满足：

$$
X_0\sim p_0,
$$

那么在理想条件下，ODE 产生的随机变量满足：

$$
X_t\sim p_t.
$$

最终得到：

$$
X_1\sim p_1=p_{\mathrm{data}}.
$$

理论生成过程可以写成：

$$
\boxed{
X_0\sim p_0
\quad\xrightarrow{\text{求解 ODE}}\quad
X_1\sim p_{\mathrm{data}}
}.
$$

---

## 九、生成实践：代码实际怎么操作？

实际中，神经网络只负责回答：

> 粒子当前位于 $x$、时间为 $t$ 时，下一瞬间应该以什么速度运动？

网络调用为：

```python
velocity = model(x, t)
```

然后使用 ODE 数值求解器反复更新状态。

### 9.1 采样初始噪声

```python
x = torch.randn(sample_shape, device=device)
```

对应：

$$
X_0\sim p_0.
$$

### 9.2 Euler 数值积分

```python
@torch.no_grad()
def sample_euler(model, shape, device, num_steps=50):
    model.eval()

    x = torch.randn(shape, device=device)
    batch_size = shape[0]
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_value = i / num_steps

        t = torch.full(
            (batch_size,),
            t_value,
            device=device,
        )

        velocity = model(x, t)
        x = x + dt * velocity

    return x
```

每一步近似：

$$
X_{t+\Delta t}
\approx
X_t+\Delta t\,v_\theta(X_t,t).
$$

实际项目还可以使用 Heun、Midpoint、Runge–Kutta 或自适应 ODE 求解器。

### 9.3 为什么实际生成只是近似？

理论中假设：

- 网络完全等于真实边缘向量场；
- ODE 被精确求解；
- 数据分布被准确表示。

实际中存在三类误差。

#### 模型误差

$$
v_\theta(x,t)\neq v_t(x).
$$

网络没有完全学到真实边缘速度。

#### 数值积分误差

Euler、Heun、Runge–Kutta 都只能近似求解 ODE。步数越少，离散误差通常越明显。

#### 数据与统计误差

训练使用有限经验数据：

$$
\hat p_{\mathrm{data}}\neq p_{\mathrm{data}}.
$$

因此最终只能期待：

$$
p_\theta(x)\approx p_{\mathrm{data}}(x).
$$

---

## 十、理论与实践为什么存在这些区别？

根本原因是高维积分和密度计算不可行。

理论边缘向量场为：

$$
v_t(x)
=
\frac{
\int p_t(x\mid z)u_t(x\mid z)q(z)\,\mathrm dz
}{p_t(x)}.
$$

对图像等高维数据而言：

- $x$ 可能有数十万维；
- $z$ 同时包含噪声端点和数据端点；
- $p_{\mathrm{data}}$ 没有可计算的解析密度；
- 积分维度极高；
- 后验 $p_t(z\mid x)$ 难以计算。

因此无法先遍历所有端点，再为每个 $x$ 显式计算平均速度。

实践采用两个替代手段。

### 10.1 Monte Carlo 采样替代理论积分

理论：

$$
\mathbb E_{Z\sim q}[f(Z)].
$$

实践：

$$
\frac1B\sum_{i=1}^{B}f(z_i).
$$

DataLoader、噪声采样器和时间采样器不断产生 Monte Carlo 样本。

### 10.2 神经网络回归替代显式条件期望

理论：

$$
v_t(x)=\mathbb E[U_t\mid X_t=x].
$$

实践：

```python
loss = mse(model(xt, t), ut)
```

神经网络通过大量训练样本，学习从 $(x,t)$ 到条件平均速度的映射。

因此可以把神经网络理解成：

> 一个经过大量样本训练后，能够快速近似高维条件期望的函数。

---

## 十一、完整信息流

```text
训练阶段
────────────────────────────────────────────

DataLoader 采样 X₁ ──┐
                     ├──> 端点条件 Z=(X₀,X₁)
高斯采样器生成 X₀  ───┘
                               ↓
                        随机采样时间 t
                               ↓
                     构造中间状态 Xₜ
                               ↓
                     计算条件速度 Uₜ
                               ↓
                MSE(model(Xₜ,t), Uₜ)
                               ↓
              网络隐式学到边缘速度 vₜ(x)


生成阶段
────────────────────────────────────────────

                       采样 X₀~p₀
                               ↓
                     输入 model(Xₜ,t)
                               ↓
                       得到当前速度
                               ↓
                     ODE 数值更新 Xₜ
                               ↓
                       重复直到 t=1
                               ↓
                  得到近似数据样本 X₁
```

---

## 十二、Rectified Flow：从理论到代码一次串联

### 12.1 理论

选择端点：

$$
X_0\sim p_0,
\qquad
X_1\sim p_{\mathrm{data}}.
$$

选择路径：

$$
X_t=(1-t)X_0+tX_1.
$$

条件速度：

$$
U_t=X_1-X_0.
$$

边缘速度：

$$
v_t(x)
=
\mathbb E[X_1-X_0\mid X_t=x].
$$

理想学习目标：

$$
v_\theta(x,t)\approx v_t(x).
$$

生成 ODE：

$$
\frac{\mathrm dX_t}{\mathrm dt}
=
v_\theta(X_t,t).
$$

### 12.2 实践训练

```python
x1 = next(dataloader).to(device)
x0 = torch.randn_like(x1)

t = torch.rand(x1.shape[0], device=device)
t_view = t.reshape(
    t.shape[0],
    *([1] * (x1.ndim - 1)),
)

xt = (1.0 - t_view) * x0 + t_view * x1
target = x1 - x0

prediction = model(xt, t)
loss = ((prediction - target) ** 2).mean()

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

它没有显式计算：

```python
q_density
p_t_density
posterior
marginal_velocity
```

但 MSE 回归会让模型逼近：

$$
v_t(x)
=
\mathbb E[U_t\mid X_t=x].
$$

### 12.3 实践生成

```python
x = torch.randn(sample_shape, device=device)
dt = 1.0 / num_steps

for i in range(num_steps):
    t = torch.full(
        (x.shape[0],),
        i / num_steps,
        device=device,
    )

    x = x + dt * model(x, t)
```

最终的 $x$ 就是生成样本。

---

## 十三、常见误解检查

### 误解 1：理论写了 $q(z)$，代码就必须计算 $q(z)$ 的密度

错误。代码通常只需要能够按照 $q$ 产生样本。

### 误解 2：理论写了后验 $p_t(z\mid x)$，训练就要先计算后验

错误。后验用于证明边缘速度的形式，MSE 回归会隐式实现后验平均。

### 误解 3：训练前必须先显式算出边缘向量场

错误。实际使用可计算的条件速度训练，网络最终学到边缘向量场。

### 误解 4：条件轨迹是直线，所以生成轨迹一定是直线

错误。条件速度经过后验平均后，边缘向量场随位置变化，生成轨迹可能弯曲。

### 误解 5：训练时需要完整求解 ODE

错误。普通 FM 训练每次只随机采样一个时间和中间状态，不需要积分完整轨迹。ODE 求解主要发生在生成阶段。

### 误解 6：CFM Loss 必须降到零

错误。同一 $(X_t,t)$ 可能对应多个不同条件速度，因此存在条件速度的不可约方差。即使网络学到了正确边缘向量场，CFM Loss 也未必为零。

---

## 十四、最后只记住四句话

### 1. 理论设计条件路径并得到条件速度

$$
X_t=I_t(Z),
\qquad
U_t=\partial_t I_t(Z).
$$

### 2. 理论证明条件速度的后验平均就是边缘向量场

$$
v_t(x)
=
\mathbb E[U_t\mid X_t=x].
$$

### 3. 实际训练不计算后验和边缘向量场，而是采样条件速度做 MSE 回归

```python
loss = mse(model(xt, t), ut)
```

### 4. 训练后的网络近似边缘向量场，生成时使用它求解 ODE

$$
\frac{\mathrm dX_t}{\mathrm dt}
=
v_\theta(X_t,t).
$$

最核心的理论—实践对应为：

$$
\boxed{
\underbrace{
v_t(x)=\mathbb E[U_t\mid X_t=x]
}_{\text{理论说明应该学习什么}}
\quad\Longleftrightarrow\quad
\underbrace{
\min_\theta
\mathbb E\|v_\theta(X_t,t)-U_t\|^2
}_{\text{实践说明如何通过采样学到它}}
}
$$

> 理论中的积分、边缘化和后验负责证明目标正确；实践中的采样、MSE 和神经网络负责在无法显式计算这些理论对象时近似实现它。
