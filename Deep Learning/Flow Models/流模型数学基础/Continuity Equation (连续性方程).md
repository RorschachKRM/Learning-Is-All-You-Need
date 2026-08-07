---
title: Continuity Equation，连续性方程
tags:
  - Flow-Model
  - Flow-Matching
  - Mathematics
  - PDE
related:
  - "[[Flow Model]]"
  - "[[Flow Matching理论与实践思维导图]]"
  - "[[ODE (Ordinary differential equations, 常微分方程)]]"
---

# Continuity Equation（连续性方程）

> [!summary] 一句话理解
> **连续性方程描述概率质量在速度场中如何流动：某处概率密度的增加，等于流入该处的概率减去流出的概率。**

> [!important] 本文范围
> 本文讨论确定性 ODE 流中的概率连续性方程
> $$
> \partial_t p_t+\nabla\!\cdot(p_tu_t)=0.
> $$
> 它描述概率守恒，不是流体的动量方程，也不包含随机扩散项。

## 1. 符号与对象

考虑时间依赖速度场：

$$
u_t(x)=u(t,x)\in\mathbb R^d,
$$

以及由它定义的 ODE：

$$
\frac{\mathrm dX_t}{\mathrm dt}=u_t(X_t).
$$

若初始位置是随机变量 $X_0\sim p_0$，那么每个时刻都有：

$$
X_t\sim p_t.
$$

其中：

| 记号 | 含义 |
|---|---|
| $X_t$ | 一个随机粒子在时刻 $t$ 的位置 |
| $p_t(x)$ | 时刻 $t$、位置 $x$ 处的概率密度 |
| $u_t(x)$ | 时刻 $t$、位置 $x$ 处的速度向量 |
| $p_t(x)u_t(x)$ | 概率通量（probability flux） |

注意：$p_t(x)$ 是概率**密度**，连续随机变量取到单个点 $x$ 的概率通常为零。

## 2. 连续性方程

$$
\boxed{
\frac{\partial p_t(x)}{\partial t}
+\nabla\!\cdot\bigl(p_t(x)u_t(x)\bigr)=0.
}
$$

两项分别表示：

- $\partial_t p_t(x)$：固定位置 $x$ 处的密度随时间怎样变化；
- $\nabla\!\cdot(p_tu_t)$：该位置附近概率通量的净流出程度。

因此：

$$
\text{局部密度变化}+\text{净流出}=0.
$$

- 净流出为正，局部密度下降；
- 净流入为正，等价于净流出为负，局部密度上升。

## 3. 积分形式：区域内的概率守恒

取一个固定空间区域 $\Omega$，其中的总概率为：

$$
\int_\Omega p_t(x)\,\mathrm dx.
$$

它的变化率等于穿过边界的负通量：

$$
\boxed{
\frac{\mathrm d}{\mathrm dt}
\int_\Omega p_t(x)\,\mathrm dx
=-
\int_{\partial\Omega}p_t(x)u_t(x)\cdot n(x)\,\mathrm dS.
}
$$

其中 $n(x)$ 是区域边界的外法向量。

- $u_t\cdot n>0$：概率向区域外流出；
- $u_t\cdot n<0$：概率向区域内流入。

利用散度定理：

$$
\int_{\partial\Omega}p_tu_t\cdot n\,\mathrm dS
=\int_\Omega\nabla\!\cdot(p_tu_t)\,\mathrm dx,
$$

即可由积分形式得到局部微分形式：

$$
\partial_t p_t+\nabla\!\cdot(p_tu_t)=0.
$$

如果无穷远处通量消失或边界没有概率流出，则：

$$
\frac{\mathrm d}{\mathrm dt}\int_{\mathbb R^d}p_t(x)\,\mathrm dx=0,
$$

所以总概率始终为 $1$。

## 4. 展开形式与沿轨迹的密度变化

使用乘积法则：

$$
\nabla\!\cdot(p_tu_t)
=u_t\cdot\nabla p_t+p_t\nabla\!\cdot u_t.
$$

连续性方程可以写成：

$$
\partial_t p_t
+u_t\cdot\nabla p_t
=-p_t\nabla\!\cdot u_t.
$$

沿 ODE 轨迹 $X_t$，链式法则给出：

$$
\frac{\mathrm d}{\mathrm dt}p_t(X_t)
=\partial_t p_t(X_t)
+u_t(X_t)\cdot\nabla p_t(X_t).
$$

因此：

$$
\boxed{
\frac{\mathrm d}{\mathrm dt}p_t(X_t)
=-p_t(X_t)\,\nabla\!\cdot u_t(X_t).
}
$$

当 $p_t(X_t)>0$ 时，两边除以密度：

$$
\boxed{
\frac{\mathrm d}{\mathrm dt}\log p_t(X_t)
=-\nabla\!\cdot u_t(X_t).
}
$$

这说明沿着一个运动粒子：

- $\nabla\!\cdot u_t>0$：周围空间膨胀，密度下降；
- $\nabla\!\cdot u_t<0$：周围空间压缩，密度上升；
- $\nabla\!\cdot u_t=0$：密度沿轨迹保持不变。

## 5. 与流映射和 Jacobian 的关系

设 ODE 的流映射为：

$$
X_t=\Phi_t(X_0).
$$

概率守恒意味着一个初始小体积经过流变形后，其中的概率质量不变：

$$
\boxed{
p_t\bigl(\Phi_t(x_0)\bigr)
\left|\det D\Phi_t(x_0)\right|
=p_0(x_0).
}
$$

等价地：

$$
p_t(x)
=p_0\bigl(\Phi_t^{-1}(x)\bigr)
\left|\det D\Phi_t^{-1}(x)\right|.
$$

Jacobian 行列式衡量局部体积如何改变：

- $|\det D\Phi_t|>1$：体积膨胀，密度降低；
- $|\det D\Phi_t|<1$：体积压缩，密度升高。

并且：

$$
\frac{\mathrm d}{\mathrm dt}
\log\left|\det D\Phi_t(x_0)\right|
=\nabla\!\cdot u_t\bigl(\Phi_t(x_0)\bigr).
$$

它与沿轨迹的对数密度变化公式符号相反，体现“体积扩大时密度减小”。

## 6. 例子一：匀速平移

在一维中令：

$$
u_t(x)=c.
$$

连续性方程变为：

$$
\partial_t p_t(x)+c\,\partial_x p_t(x)=0.
$$

ODE 轨迹为：

$$
X_t=X_0+ct,
$$

所以概率密度整体平移：

$$
\boxed{p_t(x)=p_0(x-ct).}
$$

若 $X_0\sim\mathcal N(0,1)$，则：

$$
X_t\sim\mathcal N(ct,1).
$$

此时 $\partial_xu_t=0$，空间既不膨胀也不压缩，所以分布形状和方差不变。

## 7. 例子二：向原点压缩

在一维中令：

$$
u(x)=-x.
$$

ODE 解为：

$$
X_t=e^{-t}X_0.
$$

流将空间长度缩小为原来的 $e^{-t}$，所以密度相应升高：

$$
\boxed{p_t(x)=e^t p_0(e^tx).}
$$

因为：

$$
\partial_xu=-1<0,
$$

所以沿轨迹有：

$$
\frac{\mathrm d}{\mathrm dt}\log p_t(X_t)=1.
$$

如果 $X_0\sim\mathcal N(0,1)$，那么：

$$
X_t\sim\mathcal N(0,e^{-2t}),
$$

分布逐渐向原点集中。

## 8. 在 Flow Model 中的作用

### Flow Model

速度场通过 ODE 决定粒子轨迹：

$$
\dot X_t=u_t(X_t),
$$

连续性方程则决定这些粒子的整体分布如何演化：

$$
\partial_t p_t+\nabla\!\cdot(p_tu_t)=0.
$$

两者分别描述同一个流的粒子视角和分布视角。

### Flow Matching

[[Flow Matching理论与实践思维导图]] 先选定一条概率路径 $(p_t)$，再学习能使其满足连续性方程的速度场 $u_t$：

$$
\text{选定 }p_t
\quad\Longrightarrow\quad
\text{学习满足 }\partial_t p_t+\nabla\!\cdot(p_tu_t)=0\text{ 的 }u_t.
$$

因此 Flow Matching 的核心并不是默认选择直线，而是匹配所选概率路径对应的速度场。

### Continuous Normalizing Flow

Continuous Normalizing Flow 使用：

$$
\frac{\mathrm d}{\mathrm dt}\log p_t(X_t)
=-\nabla\!\cdot u_t(X_t)
$$

计算样本对数密度的变化，并据此进行最大似然训练或密度评估。

## 9. 与 Fokker--Planck 方程的区别

确定性 ODE：

$$
\mathrm dX_t=u_t(X_t)\,\mathrm dt
$$

只产生搬运项，对应连续性方程。

如果运动中还含有随机噪声，例如 SDE：

$$
\mathrm dX_t=b_t(X_t)\,\mathrm dt
+\sigma_t(X_t)\,\mathrm dW_t,
$$

那么密度方程还会出现扩散项，得到更一般的 Fokker--Planck 方程。因此，连续性方程可以看作“没有随机扩散时”的密度演化方程。

## 10. 常见误解

### “连续性方程是一条 ODE”

不是。$p_t(x)$ 同时依赖时间 $t$ 和空间 $x$，连续性方程是偏微分方程（PDE）。单个粒子的运动方程 $\dot X_t=u_t(X_t)$ 才是 ODE。

### “$p_t$ 是粒子轨迹”

不是。$X_t$ 描述单个粒子的位置，$p_t$ 描述所有随机粒子在时刻 $t$ 的概率分布。

### “散度为零表示粒子没有运动”

不是。$\nabla\!\cdot u_t=0$ 只表示局部体积不膨胀、不压缩；粒子仍然可以平移或旋转。

### “给定概率路径后，速度场一定唯一”

一般不一定。连续性方程只约束速度场如何搬运密度，在多维空间中可能存在多个速度场产生同一概率路径；额外结构或优化准则可以帮助选定具体速度场。

## 11. 核心关系

$$
\boxed{
\begin{aligned}
\text{粒子运动：}\quad
&\dot X_t=u_t(X_t),\\
\text{分布演化：}\quad
&\partial_t p_t+\nabla\!\cdot(p_tu_t)=0,\\
\text{流的换元公式：}\quad
&p_t(\Phi_t(x_0))|\det D\Phi_t(x_0)|=p_0(x_0),\\
\text{沿轨迹的密度变化：}\quad
&\frac{\mathrm d}{\mathrm dt}\log p_t(X_t)
=-\nabla\!\cdot u_t(X_t).
\end{aligned}
}
$$
