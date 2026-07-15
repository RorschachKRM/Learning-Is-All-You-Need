---
title: python numpy数组
tags:
  - Python
  - Numpy
related:
---

# 一、NumPy 简介

NumPy（Numerical Python）是 Python 科学计算的核心库，其核心数据结构是 `ndarray`（N维数组对象）。
**为什么需要 NumPy 数组？**
与 Python 原生列表相比，NumPy 数组有以下显著优势：
- **计算效率高**：底层由 C 和 Fortran 实现，支持并行操作，速度远快于原生列表
- **内存效率高**：存储同类型数据，内存布局更紧凑
- **功能丰富**：提供大量数学函数和数组操作工具
- **生态互操作**：与 SciPy、Pandas、Matplotlib 等科学计算库完美配合


# 二、创建 NumPy 数组
## 1. 使用 `np.array()` 从列表或元组创建

最基本的方式是将 Python 序列转换为 NumPy 数组。
⚠️ 注意：`np.array()` 接收的是**一个序列**，而不是多个参数。
```python
import numpy as np

# 一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # 输出: [1 2 3 4 5]

# 二维数组（嵌套列表）
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# 输出:
# [[1 2 3]
#  [4 5 6]]

# 指定数据类型
arr3 = np.array([1, 2, 3], dtype=float)
print(arr3)  # 输出: [1. 2. 3.]
```

## 2. 使用特殊函数创建

NumPy 提供了多种便捷函数来创建特定类型的数组：

|函数|说明|示例|
|---|---|---|
|`np.zeros((3, 4))`|创建全0数组|3行4列全0|
|`np.ones((2, 3))`|创建全1数组|2行3列全1|
|`np.empty((2, 2))`|创建未初始化的数组（值随机）|2行2列|
|`np.arange(0, 10, 2)`|等差数列（类似range）|[0, 2, 4, 6, 8]|
|`np.linspace(0, 10, 5)`|等间隔数列（指定数量）|[0, 2.5, 5, 7.5, 10]|
|`np.eye(3)`|单位矩阵|3×3单位阵|
|`np.random.random((2, 2))`|随机数数组（[0,1)区间）|2×2随机数|

# 三、数组属性
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)   # (2, 3) — 2行3列
print(arr.ndim)    # 2 — 二维数组
print(arr.size)    # 6 — 元素总数
print(arr.dtype)   # int64 — 元素数据类型
```


# 四、索引与切片
## 1. 基本索引

索引从 0 开始，使用 `[行, 列]` 的方式访问多维数组元素：
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr[0, 1])   # 输出: 2（第0行第1列）
print(arr[1, 2])   # 输出: 6（第1行第2列）
```

## 2. 切片
切片语法为 `[start:stop:step]`，遵循**左闭右开**规则
⚠️ **重要**：切片返回的是**视图（view）** 而非副本，修改切片会影响原数组
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr[1, :])     # [4 5 6] — 第1行所有列
print(arr[:, 1])     # [2 5 8] — 所有行第1列
print(arr[1:3, 1:3]) # [[5 6] [8 9]] — 子矩阵

# 使用省略号 ... 保持维度一致
print(arr[..., 1])   # [2 5 8] — 所有行第1列
print(arr[1, ...])   # [4 5 6] — 第1行所有列
```


# 五、形状操作

## 1. `reshape()` — 改变形状

`reshape()` 在不改变数据的情况下调整数组形状，新形状的元素总数必须与原数组一致：
```python
arr = np.arange(8)  # [0 1 2 3 4 5 6 7]

# 重塑为 4行×2列
reshaped = arr.reshape(4, 2)
print(reshaped)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

# 重塑为 2行×2列×2列（三维）
arr_3d = arr.reshape(2, 2, 2)
```

## 2. 展平数组

```python
arr = np.array([[1, 2], [3, 4]])

print(arr.ravel())    # [1 2 3 4] — 返回视图
print(arr.flatten())  # [1 2 3 4] — 返回副本
```

## 3. 转置

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.T)  # 或 arr.transpose()
# [[1 4]
#  [2 5]
#  [3 6]]
```

# 六、数学运算与广播

## 1. 元素级运算

算术运算会作用于数组的每一个元素：
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)      # [5 7 9]
print(a * b)      # [4 10 18]
print(a ** 2)     # [1 4 9]
print(a + 10)     # [11 12 13] — 标量运算广播到每个元素
```

## 2. 通用函数（ufunc）

NumPy 提供了丰富的数学函数：
```python
arr = np.array([1, 4, 9, 16])
print(np.sqrt(arr))   # [1. 2. 3. 4.]
print(np.exp(arr))    # 指数
print(np.log(arr))    # 自然对数
print(np.abs([-1, -2, -3]))  # [1 2 3]
```

## 3. 聚合函数

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.sum())      # 21 — 所有元素求和
print(arr.mean())     # 3.5 — 平均值
print(arr.max())      # 6 — 最大值
print(arr.min())      # 1 — 最小值
print(arr.sum(axis=0))  # [5 7 9] — 按列求和
print(arr.sum(axis=1))  # [6 15] — 按行求和
```

## 4. 广播（Broadcasting）

广播允许对不同形状的数组进行运算，较小的数组会自动扩展：
```python
a = np.array([[1, 2], [3, 4]])  # 2×2
b = np.array([10, 20])          # 一维数组
print(a + b)
# [[11 22]
#  [13 24]]
# b 被自动扩展为 [[10, 20], [10, 20]]
```

## 5. 矩阵运算
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.dot(a, b))  # 矩阵乘法
# [[19 22]
#  [43 50]]
print(np.linalg.inv(a))  # 矩阵求逆
print(np.linalg.det(a))  # 行列式
```