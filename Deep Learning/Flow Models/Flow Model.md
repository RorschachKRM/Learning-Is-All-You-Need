---
title: Flow Model，连续时间生成流模型
tags:
  - Flow-Model
  - Generative-Model
  - ODE
related:
  - "[[Flow Matching]]"
  - "[[Rectified Flow]]"
  - "[[ODE (Ordinary differential equations, 常微分方程)]]"
---

# Flow Model

> [!summary] 一句话理解
> **连续时间 Flow Model 用一个速度场定义 ODE，把容易采样的源分布 $p_0$ 连续、确定性地变换成目标数据分布 $p_1$。**

> [!important] 本文范围
> 本文中的 Flow Model 专指以ODE为基础的**连续时间生成流模型**。更广义的 flow-based model 还包括由有限个可逆层组成的离散 Normalizing Flow；本文不讨论光流（optical flow），也不把 Flow Model 等同于某一种训练方法。

## 1. 模型要解决什么问题

已知：

- $p_0$：容易采样的源分布，通常是 $\mathcal N(0,I)$；
- $p_1=p_{\mathrm{data}}$：希望学习的真实数据分布。

Flow Model 希望构造一族随时间变化的分布：

$$
p_0\longrightarrow p_t\longrightarrow p_1,
\qquad t\in[0,1].
$$

它不是直接“移动概率密度图”，而是让大量样本沿速度场运动；样本整体的位置分布随之从 $p_0$ 变成 $p_1$。

也即连续时间流模型通过 ODE 搬运概率分布，所以 Flow Model 描述的是整体建模思想：
$$p_0\xrightarrow{\text{ODE flow}}p_1$$

## 2. 速度场、ODE、轨迹与流

### 2.1 速度场

$$
u_t(x)=u(t,x)\in\mathbb R^d
$$

表示：在时刻 $t$、空间位置 $x$ 的瞬时速度。它是运动规则。

### 2.2 ODE 与轨迹

给定初始位置 $X_0=x_0$，样本按照 ODE 运动：

$$
\frac{\mathrm dX_t}{\mathrm dt}=u_t(X_t),
\qquad X_0=x_0.
$$

求解得到函数 $t\mapsto X_t$。它是该初值问题的解，也是一条轨迹。

### 2.3 流映射

对每个初始点都求解 ODE，得到：

$$
X_t=\Phi_t(x_0).
$$

固定 $t$ 时，$\Phi_t$ 将每个初始位置映射到其当前的位置：

$$
\Phi_t:\mathbb R^d\to\mathbb R^d.
$$

因此：

- 一条轨迹描述一个初始点怎样运动；
- 流 $\Phi_t$ 描述所有初始点在时刻 $t$ 被整体映射到哪里。

## 3. 从移动粒子到搬运分布

如果初始位置是随机变量：

$$
X_0\sim p_0,
$$

那么：

$$
X_t=\Phi_t(X_0)
$$

仍然是随机变量，其分布记为 $p_t$。分布层面的关系写作：

$$
\boxed{p_t=(\Phi_t)_\#p_0.}
$$

$(\Phi_t)_\#p_0$ 表示流 $\Phi_t$ 将 $p_0$ 推送（pushforward）成 $p_t$。生成模型的目标是：

$$
(\Phi_1)_\#p_0=p_1.
$$

粒子层面与分布层面要区分：

| 层面 | 数学表达 | 含义 |
|---|---|---|
| 单个样本 | $x_0\mapsto\Phi_t(x_0)$ | 一个点沿轨迹运动 |
| 随机变量 | $X_t=\Phi_t(X_0)$ | 随机初值经过流变换 |
| 概率分布 | $p_t=(\Phi_t)_\#p_0$ | 整个分布被搬运 |

## 4. 概率密度如何随流变化

速度场与概率路径满足连续性方程：

$$
\frac{\partial p_t(x)}{\partial t}
+\nabla\!\cdot\bigl(p_t(x)u_t(x)\bigr)=0.
$$

它表达概率质量守恒：概率不会凭空产生或消失，只会被速度场搬运。

沿一条轨迹，密度变化满足瞬时换元公式：

$$
\frac{\mathrm d}{\mathrm dt}\log p_t(X_t)
=-\nabla\!\cdot u_t(X_t).
$$

其中 $\nabla\!\cdot u_t$ 是速度场的散度。这个公式是 Continuous Normalizing Flow 计算似然的基础。

## 5. Flow Model 本身不规定如何训练

Flow Model 只规定模型形式：

$$
\dot X_t=u_\theta(t,X_t).
$$

它没有自动给出神经网络参数 $\theta$。速度场可以使用不同目标训练：

| 方法 | 核心思想 | 是否规定直线路径 |
|---|---|---|
| Continuous Normalizing Flow 最大似然 | 使用密度变化公式优化数据似然 | 否 |
| [[Flow Matching]] | 选择概率路径，回归对应速度场 | 否 |
| [[Rectified Flow]] | 使用端点直线条件路径，并可用 Reflow 校直 | 是，针对其基础条件路径 |

所以三者的关系是：

> **Flow Model 是模型范式；Flow Matching 是通用训练框架；Rectified Flow 是采用特定直线条件路径和校直策略的具体方法。**

## 6. 训练阶段与生成阶段

### 训练阶段

目标是学习速度场参数：

$$
u_\theta(t,x).
$$

训练信号取决于所用方法。例如 Flow Matching 回归目标速度，CNF 最大似然则优化密度。

### 生成阶段

![[Pasted image 20260802170523.png]]

速度场固定后：

1. 采样 $Z_0\sim p_0$；
2. 求解 $\dot Z_t=u_\theta(t,Z_t)$；
3. 输出终点 $Z_1=\Phi_1(Z_0)$。

因此：

- 神经网络训练学习的是速度场；
- ODE 求解得到的是单个样本轨迹；
- 所有初值对应的轨迹共同定义流；
- 流把初始分布搬运到生成分布。

## 7. 一维平移例子：只说明模型机制

设已知速度场：

$$
u_t(x)=5.
$$

则 ODE 为：

$$
\frac{\mathrm dX_t}{\mathrm dt}=5,
\qquad X_0=x_0,
$$

轨迹和流分别为：

$$
X_t=x_0+5t,
\qquad
\Phi_t(x_0)=x_0+5t.
$$

如果 $X_0\sim\mathcal N(0,1)$，则：

$$
X_t=X_0+5t\sim\mathcal N(5t,1).
$$

这个例子只说明“给定速度场后，ODE 如何产生轨迹、流和概率路径”，**不表示所有 Flow Model 或 Flow Matching 都使用恒定速度或直线轨迹**。

## 8. 可逆性与基本条件

如果速度场对空间变量足够规则，例如满足适当的 Lipschitz 条件，则给定初值的 ODE 解存在且唯一。此时不同轨迹不会在同一时刻相交，流映射通常是可逆的。

对于依赖时间的速度场，更完整的流可写为：

$$
\Phi_{t,s}(x),
$$

表示把时刻 $s$ 的位置 $x$ 映射到时刻 $t$。常写的 $\Phi_t(x_0)$ 相当于固定初始时刻为 $0$：

$$
\Phi_t(x_0)=\Phi_{t,0}(x_0).
$$

## 9. 概念边界

| 概念             | 已知/给定              | 要得到的对象             |
| -------------- | ------------------ | ------------------ |
| ODE 初值问题       | 速度场 $u_t$、初值 $x_0$ | 一条轨迹 $X_t$         |
| 流              | 速度场 $u_t$、所有可能初值   | 映射族 $\Phi_t$       |
| Flow Model 训练  | 数据、基础分布、训练目标       | 参数化速度场 $u_\theta$  |
| Flow Model 生成  | 已训练速度场、随机初值 $Z_0$  | 生成轨迹及终点 $Z_1$      |
| Flow Matching  | 选定概率路径及其速度监督       | 拟合该路径的速度场          |
| Rectified Flow | 端点耦合和直线条件路径        | RF 速度场，可进一步 Reflow |

## 10. 核心公式

$$
\boxed{
\begin{aligned}
\text{运动规则：}\quad
&\dot X_t=u_\theta(t,X_t),\\
\text{流映射：}\quad
&X_t=\Phi_t(X_0),\\
\text{分布搬运：}\quad
&p_t=(\Phi_t)_\#p_0,\\
\text{生成目标：}\quad
&(\Phi_1)_\#p_0=p_1.
\end{aligned}
}
$$

