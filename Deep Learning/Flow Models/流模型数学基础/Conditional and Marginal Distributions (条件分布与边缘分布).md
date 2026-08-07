---
title: Conditional and Marginal Distributions，条件分布与边缘分布
tags:
  - Probability
  - Flow-Matching
  - Mathematics
related:
  - "[[Flow Matching理论与实践思维导图]]"
  - "[[Continuity Equation (连续性方程)]]"
  - "[[Flow Model]]"
---

# 条件分布与边缘分布

> [!summary] 一句话理解
> **条件对象描述“已知样本来自哪个条件时会怎样”；边缘对象则把条件变量隐藏并加权混合，描述所有条件共同形成的总体。**

> [!important] 本文范围
> 本文只整理理解 [[Flow Matching理论与实践思维导图]] 所需的条件分布、边缘分布、概率路径和速度场概念。“边缘”是 marginal 的翻译，表示对其他变量求和或积分，并不是空间边界。

## 1. 联合、条件与边缘分布

设：

- $C$：条件变量，可以代表类别、数据样本或一对端点；
- $X$：关心的随机变量；
- $q(c)$：条件变量 $C$ 的分布。

### 联合分布

$$
p(x,c)
$$

描述 $X=x$ 和 $C=c$ 同时出现的可能性。

### 条件分布

$$
p(x\mid c)
$$

描述已经知道 $C=c$ 后，$X$ 如何分布。

联合分布可以分解为：

$$
\boxed{p(x,c)=p(x\mid c)q(c).}
$$

### 边缘分布

如果不再关心 $C$ 的具体取值，就把它求和或积分掉：

离散条件：

$$
p(x)=\sum_c p(x\mid c)q(c).
$$

连续条件：

$$
\boxed{
p(x)=\int p(x\mid c)q(c)\,\mathrm dc.
}
$$

这个操作叫**边缘化（marginalization）**。得到的 $p(x)$ 是所有条件分布按 $q(c)$ 加权混合后的总体分布。

## 2. 一个简单的混合例子

假设人群来自两个组：

$$
P(C=A)=0.7,
\qquad
P(C=B)=0.3.
$$

两个组分别有条件分布：

$$
p(x\mid A),
\qquad
p(x\mid B).
$$

如果不知道一个样本来自哪个组，它的总体分布就是：

$$
p(x)=0.7p(x\mid A)+0.3p(x\mid B).
$$

因此：

- 固定 $C=A$ 观察到的是条件分布；
- 忽略组别、观察全体得到的是边缘分布。

## 3. 条件概率路径

在 Flow Matching 中，分布还依赖时间 $t$。固定条件 $C=c$ 后：

$$
\bigl(p_t(x\mid c)\bigr)_{t\in[0,1]}
$$

称为一条**条件概率路径**。

它表示：

> 已知条件为 $c$ 时，这一组粒子的概率分布如何随时间变化。

条件 $C$ 可以是：

- 一个目标数据样本 $X_1$；
- 一对端点 $(X_0,X_1)$；
- 类别标签或其他条件信息。

条件概率路径不要求是直线。其具体形状取决于所选的路径构造。

## 4. 边缘概率路径

在每个时刻把所有条件路径混合：

$$
\boxed{
p_t(x)=\int p_t(x\mid c)q(c)\,\mathrm dc.
}
$$

当 $t$ 从 $0$ 变化到 $1$ 时：

$$
\bigl(p_t(x)\bigr)_{t\in[0,1]}
$$

称为**边缘概率路径**，也可以直接称为总体概率路径：

$$
p_0\longrightarrow p_t\longrightarrow p_1.
$$

可以这样理解：

```text
条件路径 1  ─┐
条件路径 2  ─┼─→ 在每个时刻混合 ─→ 边缘概率路径 pₜ
条件路径 3  ─┘
```

边缘概率路径描述的是整个粒子群如何分布，不是一个样本在空间中的运动轨迹。

## 5. 条件速度场

每条条件概率路径都有相应的条件速度场：

$$
u_t(x\mid c).
$$

它表示：

> 已知粒子来自条件 $c$ 时，它在时刻 $t$、位置 $x$ 应该具有的速度。

条件密度和条件速度满足条件连续性方程：

$$
\partial_t p_t(x\mid c)
+\nabla\!\cdot
\left(p_t(x\mid c)u_t(x\mid c)\right)=0.
$$

在 Conditional Flow Matching 中，条件路径被设计得足够简单，使 $u_t(x\mid c)$ 可以直接计算并作为训练标签。

## 6. 边缘速度场

将所有条件概率通量混合：

$$
p_t(x)u_t(x)
=\int
p_t(x\mid c)u_t(x\mid c)q(c)\,\mathrm dc.
$$

两边除以 $p_t(x)$，得到边缘速度场：

$$
\boxed{
u_t(x)
=\frac{
\int p_t(x\mid c)u_t(x\mid c)q(c)\,\mathrm dc
}{p_t(x)}.
}
$$

它也可以写成条件期望：

$$
\boxed{
u_t(x)
=\mathbb E\!\left[
u_t(X_t\mid C)\mid X_t=x
\right].
}
$$

这不是对所有条件做简单平均，而是只考虑能够在时刻 $t$ 到达位置 $x$ 的条件，并按照它们的后验可能性加权。

边缘概率路径和边缘速度场满足：

$$
\partial_t p_t(x)
+\nabla\!\cdot\bigl(p_t(x)u_t(x)\bigr)=0.
$$

详见 [[Continuity Equation (连续性方程)]]。

## 7. 端点插值中的含义

令条件变量是一对端点：

$$
C=(X_0,X_1).
$$

如果选择插值：

$$
\widetilde X_t=I_t(X_0,X_1),
$$

则固定端点对后，条件目标速度为：

$$
V_t=\frac{\partial}{\partial t}I_t(X_0,X_1).
$$

训练中采样许多不同的端点对。固定时间 $t$，所有中间点 $\widetilde X_t$ 共同形成边缘分布 $p_t$；所有到达同一 $(t,x)$ 附近的目标速度经过条件加权平均，形成边缘速度场 $u_t(x)$。

> [!note] 直线只是特殊情况
> 如果 $I_t(X_0,X_1)=(1-t)X_0+tX_1$，条件速度为 $X_1-X_0$，这属于线性 Conditional Flow Matching，并对应基础 [[Rectified Flow]]。一般 Flow Matching 不限定 $I_t$ 必须是直线。

## 8. 路径相交时会发生什么

假设在同一个 $(t,x)$：

- $70\%$ 的相关条件给出速度 $+1$；
- $30\%$ 的相关条件给出速度 $-2$。

那么边缘速度为：

$$
u_t(x)=0.7\times1+0.3\times(-2)=0.1.
$$

虽然没有一条条件路径的速度是 $0.1$，但只知道 $(t,x)$、不知道条件来源时，均方误差意义下的最佳预测就是这个条件平均。

这也解释了：即使训练使用直线条件路径，边缘速度场产生的 ODE 轨迹仍可能弯曲。

## 9. 为什么 CFM 能学到边缘速度场

Conditional Flow Matching 使用损失：

$$
\mathcal L_{\mathrm{CFM}}
=\mathbb E
\left[
\left\|u_\theta(t,X_t)-u_t(X_t\mid C)\right\|^2
\right].
$$

网络输入只有 $(t,X_t)$，没有条件 $C$。对于均方误差，最优预测是条件期望：

$$
u_\theta^*(t,x)
=\mathbb E
\left[u_t(X_t\mid C)\mid X_t=x\right]
=u_t(x).
$$

所以训练时可以使用容易计算的条件速度作为标签，最终网络学到的却是搬运总体概率路径所需的边缘速度场。

## 10. 概念对照

| 条件层次 | 边缘层次 |
|---|---|
| 固定 $C=c$ | 将 $C$ 求和或积分掉 |
| $p_t(x\mid c)$ | $p_t(x)$ |
| 条件概率路径 | 所有条件路径混合后的总体概率路径 |
| $u_t(x\mid c)$ | $u_t(x)$ |
| 提供容易计算的训练监督 | 真正用于生成 ODE 的总体速度场 |

## 11. 常见误解

### “边缘”表示空间边界

不是。这里的边缘来自 marginal，意思是把其他变量求和或积分掉。

### 边缘分布是条件分布的简单平均

不一定。它按照条件出现的概率 $q(c)$ 加权；不同条件的权重可以不同。

### 边缘概率路径是一条样本轨迹

不是。样本轨迹是 $t\mapsto X_t$，边缘概率路径是 $t\mapsto p_t$，描述总体分布的变化。

### 边缘速度等于某一条条件路径的速度

不一定。多个条件可能在相同 $(t,x)$ 给出不同速度，边缘速度是它们的条件加权平均。

## 12. 核心关系

$$
\boxed{
\begin{aligned}
\text{边缘密度：}\quad
&p_t(x)=\int p_t(x\mid c)q(c)\,\mathrm dc,\\
\text{边缘通量：}\quad
&p_t(x)u_t(x)
=\int p_t(x\mid c)u_t(x\mid c)q(c)\,\mathrm dc,\\
\text{边缘速度：}\quad
&u_t(x)=\mathbb E[u_t(X_t\mid C)\mid X_t=x].
\end{aligned}
}
$$

