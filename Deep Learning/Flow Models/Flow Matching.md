---
title: 流匹配
tags:
  - Flow-Model
  - Flow-Matching
  - ODE
related:
  - "[[Flow Model]]"
  - "[[Rectified Flow]]"
  - "[[ODE (Ordinary differential equations, 常微分方程)]]"
---

# Flow Matching

> [!summary] 一句话理解
> **Flow Matching（FM）是一种训练连续时间 [[Flow Model]] 的通用方法：先指定一条连接 $p_0$ 与 $p_1$ 的概率路径，再用回归让神经网络匹配这条路径对应的速度场。**

## 概率路径
**概率路径是一族随时间变化的概率分布 $\{p_t \}_t∈[0,1]$。**  
它描述了从初始的简单分布 $p_0$（如标准高斯）开始，经过一系列连续的“变形”，最终到达目标分布 $p_1​$（真实数据分布）的整个过程。

> [!important] 范围说明
> Flow Matching **不规定概率路径必须是直线**。直线插值只是一个特殊选择；使用端点直线插值并强调轨迹校直的方法见 [[Rectified Flow]]。

## 1. 问题设定

设：

- $p_0$：容易采样的源分布，例如标准高斯分布；
- $p_1=p_{\mathrm{data}}$：目标数据分布；
- $(p_t)_{t\in[0,1]}$：人为选定的中间概率路径，满足 $p_{t=0}=p_0$、$p_{t=1}=p_1$；
- $u_\theta(t,x)$：神经网络表示的时间依赖速度场。
 
训练完成后，通过 ODE 生成样本：

$$
\frac{\mathrm d Z_t}{\mathrm dt}=u_\theta(t,Z_t),
\qquad Z_0\sim p_0.
$$

目标是使 $Z_t\sim p_t$，特别是 $Z_1\sim p_1$。

## 2. Flow Matching 的定义：匹配边缘速度场

概率密度 $p_t(x)$ 与搬运它的速度场 $u_t(x)$ 满足**连续性方程**：

$$
\frac{\partial p_t(x)}{\partial t}
+\nabla\!\cdot\bigl(p_t(x)u_t(x)\bigr)=0.
$$

如果 $p_t$ 及其速度场 $u_t$ 已知，理想的 Flow Matching 损失是：

$$
\boxed{
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t\sim U[0,1],\,X_t\sim p_t}
\left[
\left\|u_\theta(t,X_t)-u_t(X_t)\right\|^2
\right].
}
$$

这里匹配的是整个中间分布 $p_t$ 的**边缘速度场**。困难在于：复杂数据分布对应的 $u_t(x)$ 通常无法直接计算。

## 3. Conditional Flow Matching：可计算的实现

实践中常把复杂的边缘路径拆成许多简单的条件路径。设条件变量为 $C$，已知：

$$
p_t(x\mid C),
\qquad
u_t(x\mid C).
$$

它们混合得到边缘概率路径：

$$
p_t(x)=\int p_t(x\mid c)q(c)\,\mathrm dc.
$$

使用容易计算的条件速度训练：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t,C,\,X_t\sim p_t(\cdot\mid C)}
\left[
\left\|u_\theta(t,X_t)-u_t(X_t\mid C)\right\|^2
\right].
}
$$

均方误差的最优解是条件平均：

$$
u^*(t,x)
=\mathbb E\!\left[u_t(X_t\mid C)\mid X_t=x\right].
$$

在适当正则条件下，它正是搬运边缘分布 $p_t$ 所需的速度场。因此可以用容易监督的条件速度，间接学到难以直接计算的边缘速度场。

## 4. 一种常用构造：随机插值

一种实现 CFM 的方式是采样端点：

$$
X_0\sim p_0,
\qquad
X_1\sim p_1,
$$

再选择满足端点条件的插值函数：

$$
\widetilde X_t=I_t(X_0,X_1),
$$

$$
I_0(X_0,X_1)=X_0,
\qquad
I_1(X_0,X_1)=X_1.
$$

条件目标速度可以直接求导：

$$
V_t=\frac{\partial}{\partial t}I_t(X_0,X_1).
$$

训练目标为：

$$
\mathbb E
\left[
\left\|u_\theta(t,\widetilde X_t)-V_t\right\|^2
\right].
$$

这里 $I_t$ 可以是直线，也可以是非线性调度、扩散式路径、最优传输路径或其他可采样路径。**选择哪条路径是建模设计，不是 Flow Matching 定义的一部分。**

## 5. 一般概率路径示例：Gaussian Probability Path

令 $\varepsilon\sim\mathcal N(0,I)$、$X_1\sim p_1$，构造：

$$
\widetilde X_t=\alpha_tX_1+\sigma_t\varepsilon,
$$

其中边界通常取为：

$$
\alpha_0=0,\quad\sigma_0=1,
\qquad
\alpha_1=1,\quad\sigma_1\approx0.
$$

于是 $t=0$ 时接近标准高斯，$t=1$ 时接近数据。条件速度为：

$$
V_t=\dot\alpha_tX_1+\dot\sigma_t\varepsilon.
$$

训练时只需采样 $(\varepsilon,X_1,t)$，即可构造 $\widetilde X_t$ 和 $V_t$。不同的 $\alpha_t,\sigma_t$ 会产生不同的概率路径；它们不必对应匀速直线运动。

## 6. 训练与生成

### 训练：回归条件速度

```python
repeat:
    condition = sample_condition()
    t = Uniform(0, 1)
    xt, target_velocity = sample_conditional_path(condition, t)

    prediction = u_theta(t, xt)
    loss = mean(||prediction - target_velocity||^2)
    update(theta)
```

Flow Matching 训练通常不需要在每一步中求解 ODE。

### 生成：求解学到的 ODE

```python
z0 ~ p0
solve dz/dt = u_theta(t, z), from t=0 to t=1
return z1
```

最简单的 Euler 更新为：

$$
Z_{t+\Delta t}
\approx Z_t+\Delta t\,u_\theta(t,Z_t).
$$

也可以使用 Heun、Runge--Kutta 等 ODE 求解器。

## 7. 特殊情况：线性 Conditional Flow Matching

如果明确选择：

$$
\widetilde X_t=(1-t)X_0+tX_1,
$$

那么：

$$
V_t=X_1-X_0.
$$

对应损失为：

$$
\mathbb E
\left[
\left\|
u_\theta\!\left(t,(1-t)X_0+tX_1\right)-(X_1-X_0)
\right\|^2
\right].
$$

> [!warning] 这是特例，不是 Flow Matching 的一般定义
> 这一目标与基础 [[Rectified Flow]] 的核心训练目标相同。Rectified Flow 还特别研究端点耦合、轨迹校直和 Reflow。

## 8. 训练路径与生成轨迹的区别

- $\widetilde X_t$：训练时从条件路径直接采样的中间点；
- $Z_t$：生成时求解神经 ODE 得到的状态。

网络学习的是给定 $(t,x)$ 后的条件平均速度，因此 $Z_t$ 不一定复现某一条具体训练路径。即使使用直线条件路径，生成轨迹也可能弯曲。

## 9. 概念边界

| 概念 | 它规定什么 | 它不规定什么 |
|---|---|---|
| [[Flow Model]] | 用速度场和 ODE 搬运分布的模型形式 | 速度场如何训练 |
| Flow Matching | 匹配指定概率路径对应的速度场 | 概率路径必须是直线 |
| Conditional Flow Matching | 用可计算的条件速度训练边缘速度场 | 只能使用端点线性插值 |
| [[Rectified Flow]] | 端点直线插值，以及可选的 Reflow 校直 | 所有 Flow Matching 都必须采用它 |

## 10. 核心公式

$$
\boxed{
\begin{aligned}
\text{选择概率路径：}\quad &p_0\longrightarrow p_t\longrightarrow p_1,\\
\text{条件速度回归：}\quad
&u_\theta(t,X_t)\approx u_t(X_t\mid C),\\
\text{生成 ODE：}\quad
&\dot Z_t=u_\theta(t,Z_t),\quad Z_0\sim p_0.
\end{aligned}
}
$$

## 参考

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
