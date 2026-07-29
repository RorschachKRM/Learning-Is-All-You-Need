---
tags:
  - ODE
  - Flow-Matching
  - Mathematics
---
sources:
[什么是基于常微分方程的神经网络(Neural ODE)?如何理解它？](https://www.bilibili.com/video/BV1hk4y1e7ZK/?spm_id_from=333.337.search-card.all.click&vd_source=a196420c957ee38ebe1222999e8bedaa)
[常微分方程 (Ordinary differential equations,ODE)- MIT Diffusion 公开课笔记(一)](https://zhuanlan.zhihu.com/p/1946339771698901869)



# 1. 概念 & 定义
## 1.1 微分方程
是包含未知函数及其导数的方程，未知函数导数的最高阶数称为该微分方程的阶

## 1.2 常微分方程
是未知函数只含有一个自变量的微分方程。如：f'(α)-7f(α)= 0




# 基于常微分方程的神经网络(Neural ODE)

## 普通神经网络 & 基于ODE的网络

![[Pasted image 20260729092404.png]]

## 用ODE表示有什么优势
1. Powerful representation：微分方程可以用数值法求解，因次对于任何连续函数都有良好的逼近能力
2. Memoryefficiency：不需用到反向传播，因此训练上节约内存
3. Simplicity：不需要考虑复杂的调参和网络设计，形式简洁
4. Abstraction：让网络不需要考虑每层需要做什么，只需要考虑怎么计算结果


## 求解微分方程

![[Pasted image 20260729100014.png]]

这种解法微分方程满足一定的形式，但实际生活当中原函数比较复杂
通常会使用**数值法**求解原函数在各个点的值。比较出名的两个方法如：
	欧拉法（Euler Method)
	Runge-Kutta法（RK4）


## 欧拉法

![[Pasted image 20260729100150.png]]


### 残差块与欧拉法

![[Pasted image 20260729101541.png]]



### 欧拉法建模

![[Pasted image 20260729101610.png]]


### Euler ODE solver

![[Pasted image 20260729101408.png]]

