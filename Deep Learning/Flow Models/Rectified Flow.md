---
title: Rectified Flow，线性插值与轨迹校直
tags:
  - Flow-Model
  - Flow-Matching
  - Rectified-Flow
related:
  - "[[Flow Model]]"
  - "[[Flow Matching]]"
  - "[[ODE (Ordinary differential equations, 常微分方程)]]"
---

# Rectified Flow

> [!summary] 一句话理解
> **Rectified Flow（RF）在成对端点之间构造直线条件路径，用速度回归学习 ODE；Reflow 再利用当前流重新配对端点，使生成轨迹趋向更直、更容易数值求解。**

> [!important] 范围说明
> Rectified Flow 是一种具体方法，不是 [[Flow Matching]] 的定义。“直线”首先指训练时每对端点之间的**条件插值路径**；学到的边缘速度场及其生成轨迹不一定是直线。

## 1. 与 Flow Model、Flow Matching 的关系

| 层级 | 作用 |
|---|---|
| [[Flow Model]] | 规定用速度场与 ODE 将 $p_0$ 搬运到 $p_1$ |
| [[Flow Matching]] | 提供匹配指定概率路径速度场的通用训练框架 |
| Rectified Flow | 选用端点直线条件路径，并进一步研究耦合与 Reflow |

基础 RF 的速度回归损失与“线性 Conditional Flow Matching”相同，但 RF 额外强调运输耦合和轨迹校直。

## 2. 基础 Rectified Flow

### 2.1 选择端点耦合

从联合分布（耦合）中采样：

$$
(X_0,X_1)\sim\pi,
$$

要求其边缘分布满足：

$$
X_0\sim p_0,
\qquad
X_1\sim p_1.
$$

最简单的是独立耦合：

$$
\pi=p_0\otimes p_1,
$$

即独立采样一个噪声和一个真实数据样本。RF 也可以使用其他耦合；端点如何配对会影响路径交叉程度和训练难度。

### 2.2 构造直线条件路径

采样 $t\sim U[0,1]$，构造：

$$
\widetilde X_t=(1-t)X_0+tX_1.
$$

它连接两个端点：

$$
\widetilde X_0=X_0,
\qquad
\widetilde X_1=X_1.
$$

对时间求导得到每一对端点上的恒定条件速度：

$$
V_t=\frac{\mathrm d\widetilde X_t}{\mathrm dt}=X_1-X_0.
$$

### 2.3 回归条件速度

用神经网络 $v_\theta(t,x)$ 表示速度场：

$$
\boxed{
\mathcal L_{\mathrm{RF}}(\theta)
=\mathbb E_{(X_0,X_1)\sim\pi,\,t}
\left[
\left\|v_\theta(t,\widetilde X_t)-(X_1-X_0)\right\|^2
\right].
}
$$

训练 1-RF 时只需构造中间点并回归速度，通常不需要在训练循环中求解 ODE。

## 3. “直线”不等于学到恒定速度场

对固定端点对 $(X_0,X_1)$，条件目标速度 $X_1-X_0$ 沿该线段保持不变。但是不同端点对的方向和大小不同，所以整体速度场一般仍依赖于 $(t,x)$。

均方误差的最优边缘速度为：

$$
v^*(t,x)
=\mathbb E\!\left[X_1-X_0\mid \widetilde X_t=x\right].
$$

如果多条训练线段在同一 $(t,x)$ 附近相交，网络会平均它们的目标速度。生成轨迹则由 ODE 决定：

$$
\frac{\mathrm dZ_t}{\mathrm dt}=v_\theta(t,Z_t),
\qquad Z_0\sim p_0.
$$

因此必须区分：

| 对象 | 含义 | 是否保证为直线 |
|---|---|---|
| $\widetilde X_t$ | 某对训练端点间的人为插值 | 是 |
| $v_\theta(t,x)$ | 所有训练对共同学习出的边缘速度场 | 否 |
| $Z_t$ | 求解学习后 ODE 得到的生成轨迹 | 否 |

## 4. 为什么需要“校直”

独立配对时，训练线段可能大量交叉：

```text
X₀¹  ╲        ╱  X₁²
      ╲      ╱
       ╲    ╱
        ╳
       ╱ ╲
X₀²  ╱     ╲     X₁¹
```

交叉位置附近存在互相冲突的速度监督，条件平均可能使生成轨迹弯曲。轨迹越弯，数值积分通常需要越多步。

让轨迹更直的主要收益是：

- 可以增大 ODE 积分步长；
- 减少函数评估次数（NFE）；
- 降低少步采样的离散化误差；
- 为一步或极少步生成创造条件。

## 5. 一阶 Rectified Flow（1-RF）

从初始耦合（通常是独立耦合）直接训练得到的模型，常称为 1-RF：

```python
repeat:
    x0 ~ p0
    x1 ~ pdata
    t  ~ Uniform(0, 1)

    xt = (1 - t) * x0 + t * x1
    target = x1 - x0

    loss = mean(||v_theta(t, xt) - target||^2)
    update(theta)
```

生成时才求解 ODE：

```python
z0 ~ p0
solve dz/dt = v_theta(t, z), from t=0 to t=1
return z1
```

## 6. Reflow：利用当前流重新配对

第一次训练后，从同一个初始噪声 $Z_0$ 出发，用 1-RF 生成对应终点：

$$
\widehat Z_1=\Phi^{(1)}_1(Z_0).
$$

由此得到新耦合：

$$
(Z_0,\widehat Z_1).
$$

然后在新端点之间重新构造直线：

$$
\widetilde Z_t=(1-t)Z_0+t\widehat Z_1,
$$

并回归目标速度：

$$
\widehat Z_1-Z_0.
$$

这一再训练过程称为 **Reflow**：

```text
初始随机配对
   ↓ 训练 1-RF
用当前 ODE 生成一一对应的端点
   ↓ 重新构造直线并训练
速度冲突减少，生成轨迹趋向更直
```

常见约定是：初次训练得到 1-RF，进行一次 Reflow 后得到 2-RF。不同资料的编号表述可能略有差异。

Reflow 倾向于校直轨迹，但不保证一次后完全成为直线；生成新配对还需要额外运行已有 ODE。

## 7. 一维平移示例

若端点按确定关系配对：

$$
X_1=X_0+5,
$$

则：

$$
\widetilde X_t=X_0+5t,
\qquad
X_1-X_0=5.
$$

所有端点对的目标速度一致，网络可以学到：

$$
v_\theta(t,x)=5.
$$

此时生成轨迹也是直线：

$$
Z_t=Z_0+5t.
$$

这个例子已经没有速度冲突，因此不需要 Reflow。它只是 RF 的简单特例，不能代表一般 Flow Matching 都使用恒定速度或直线路径。

## 8. 与一般 Flow Matching 的边界

| 对比项 | Flow Matching | Rectified Flow |
|---|---|---|
| 定位 | 通用速度场训练框架 | 具体的线性路径与校直方法 |
| 概率路径 | 可以是扩散、Gaussian、OT、随机插值等 | 基础形式使用端点直线插值 |
| 条件目标速度 | 由所选条件路径决定 | $X_1-X_0$ |
| 端点耦合 | 取决于具体实现 | 明确关注耦合，并可用 Reflow 更新 |
| Reflow | 不是必要组成 | RF 的典型校直步骤 |
| 生成方式 | 求解学习到的 ODE | 求解学习到的 ODE |

## 9. 常见误解

### “Flow Matching 就是直线插值”

错误。直线插值属于线性 Conditional Flow Matching，也是基础 RF 的选择；一般 FM 不限定概率路径形式。

### “RF 学到的是处处恒定的速度场”

错误。只有每个固定端点对的条件目标速度是恒定的；不同端点对不同，最终边缘速度场通常依赖 $t$ 和 $x$。

### “训练线段是直的，所以生成轨迹一定是直的”

错误。训练线段可能相交，网络学习条件平均速度，生成 ODE 轨迹仍可能弯曲。

### “RF 直接学习流映射 $\Phi_t$”

通常不是。网络直接学习速度场 $v_\theta(t,x)$，流映射由求解 ODE 得到。

## 10. 核心公式

$$
\boxed{
\begin{aligned}
\text{直线条件路径：}\quad
&\widetilde X_t=(1-t)X_0+tX_1,\\
\text{每对端点的条件速度：}\quad
&V_t=X_1-X_0,\\
\text{速度回归：}\quad
&v_\theta(t,\widetilde X_t)\approx X_1-X_0,\\
\text{生成 ODE：}\quad
&\dot Z_t=v_\theta(t,Z_t),\\
\text{Reflow 配对：}\quad
&(Z_0,\Phi_1(Z_0)).
\end{aligned}
}
$$

## 参考

- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023.

