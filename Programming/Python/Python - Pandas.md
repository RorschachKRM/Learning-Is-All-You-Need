---
title: python pandas包
tags:
  - Python
  - Pandas
related:
---

# 一、Pandas 简介

Pandas 是 Python 数据分析的核心库，基于 NumPy 构建，提供了两个核心数据结构：

| 数据结构 | 说明 | 类比 |
|---------|------|------|
| `Series` | 一维标签数组 | Excel 的一列 |
| `DataFrame` | 二维标签数据表 | Excel 的一个工作表 |

**为什么用 Pandas？**
- **灵活的数据处理**：轻松处理缺失值、重复值、异常值
- **强大的读写能力**：支持 CSV、Excel、SQL、JSON 等多种格式
- **高效的分组聚合**：类似 SQL 的 groupby + 聚合
- **时间序列处理**：内置日期范围生成、重采样等功能
- **与其他库配合**：与 NumPy、Matplotlib、Scikit-learn 无缝衔接

```python
import pandas as pd
import numpy as np
```

# 二、创建 Series 与 DataFrame

## 1. 创建 Series

```python
# 从列表创建
s1 = pd.Series([10, 20, 30, 40])
print(s1)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# 自定义索引
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s2)
# a    10
# b    20
# c    30

# 从字典创建
s3 = pd.Series({'北京': 2154, '上海': 2487, '广州': 1868})
```

## 2. 创建 DataFrame

| 方法 | 说明 | 常用场景 |
|------|------|---------|
| `pd.DataFrame(dict)` | 从字典创建，key 为列名 | 手动构造小数据 |
| `pd.DataFrame(list_of_dicts)` | 从字典列表创建，每个 dict 为一行 | API 返回数据 |
| `pd.DataFrame(np.array)` | 从 NumPy 数组创建 | 科学计算结果 |
| `pd.read_csv()` | 从 CSV 文件读取 | 最常见的数据导入 |

```python
# 方法一：从字典创建（最常用）
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 28],
    '城市': ['北京', '上海', '广州']
})
print(df)
#    姓名  年龄  城市
# 0  张三  25  北京
# 1  李四  30  上海
# 2  王五  28  广州

# 方法二：从字典列表创建
data = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 92},
    {'name': 'Charlie', 'score': 78}
]
df2 = pd.DataFrame(data)

# 方法三：指定行索引
df3 = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6]
}, index=['row_a', 'row_b', 'row_c'])
```

# 三、读取与写入数据

## 1. 读取数据

| 函数 | 说明 |
|------|------|
| `pd.read_csv('file.csv')` | 读取 CSV 文件 |
| `pd.read_excel('file.xlsx')` | 读取 Excel 文件 |
| `pd.read_json('file.json')` | 读取 JSON 文件 |
| `pd.read_sql(query, connection)` | 从 SQL 数据库读取 |
| `pd.read_clipboard()` | 从剪贴板读取（超实用！） |

```python
# 读取 CSV 常用参数
df = pd.read_csv(
    'data.csv',
    sep=',',              # 分隔符，默认逗号
    header=0,             # 列名所在行，0 表示第一行
    index_col=0,          # 将第 0 列设为行索引
    encoding='utf-8',     # 编码
    usecols=['name', 'age'],  # 只读取指定列
    nrows=100,            # 只读取前 100 行
    skiprows=10,          # 跳过前 10 行
)

# 读取 Excel（需要 openpyxl 或 xlrd）
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

## 2. 写入数据

```python
# 写入 CSV
df.to_csv('output.csv', index=False, encoding='utf-8-sig')

# 写入 Excel
df.to_excel('output.xlsx', sheet_name='结果', index=False)

# 写入多个 Sheet
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

# 四、数据探索与基本信息

拿到数据后第一步，快速了解数据全貌：

```python
df.head(10)        # 查看前 10 行
df.tail(5)         # 查看后 5 行
df.shape           # (行数, 列数)
df.info()          # 每列的数据类型、非空数量、内存占用
df.describe()      # 数值列的统计摘要（均值、标准差、四分位数等）
df.columns         # 所有列名
df.index           # 行索引
df.dtypes          # 每列的数据类型
df.nunique()       # 每列的唯一值数量
df.memory_usage()  # 每列内存占用
```

⚠️ **重要**：`df.info()` 和 `df.describe()` 是数据探索的第一步，能帮你快速发现缺失值、异常值、数据类型问题。

# 五、索引与选择数据

Pandas 有多种数据选择方式，新手最容易混淆的就是 `loc` 和 `iloc`。

| 方式 | 依据 | 包含结尾 | 示例 |
|------|------|---------|------|
| `df['列名']` | 列标签 | — | 取单列 |
| `df[['col1', 'col2']]` | 列标签列表 | — | 取多列 |
| `df.loc[行标签, 列标签]` | **标签名** | ✅ 包含 | `df.loc['a':'c', 'name':'age']` |
| `df.iloc[行位置, 列位置]` | **整数位置** | ❌ 不包含 | `df.iloc[0:3, 0:2]` |
| `df.at[标签, 标签]` | 单个标签 | — | 取单个值（快） |
| `df.iat[位置, 位置]` | 单个位置 | — | 取单个值（快） |

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'city': ['BJ', 'SH', 'GZ', 'SZ']
}, index=['a', 'b', 'c', 'd'])

# --- 列选择 ---
df['name']                 # 返回 Series
df[['name', 'age']]        # 返回 DataFrame（注意双层中括号）

# --- loc：按标签名 ---
df.loc['b']                # 行标签为 'b' 的行（Series）
df.loc['a':'c']            # 标签 'a' 到 'c' 的行（包含 'c'！）
df.loc['a':'c', 'name':'age']  # 同时筛选行和列（都包含结尾）

# --- iloc：按整数位置 ---
df.iloc[0]                 # 第 0 行
df.iloc[0:3]               # 第 0 至 2 行（不包含第 3 行）
df.iloc[0:3, 0:2]          # 前 3 行，前 2 列

# --- 条件筛选（布尔索引）---
df[df['age'] > 28]                          # 年龄大于 28 的行
df[(df['age'] > 25) & (df['city'] == 'BJ')] # 多条件用 &（与）、|（或）
df[df['city'].isin(['BJ', 'SH'])]           # city 在列表中
df.query('age > 25 and city == "BJ"')       # query 方法（更可读）
```

⚠️ **核心区别**：`loc` 用**标签名**且**包含**结尾；`iloc` 用**整数位置**且**不包含**结尾（和 Python 切片一致）。

# 六、数据清洗

## 1. 缺失值处理

```python
# 检测
df.isnull()          # 返回布尔型 DataFrame，True 表示缺失
df.isnull().sum()    # 每列缺失值数量
df.notnull()         # 与 isnull 相反

# 删除
df.dropna()                    # 删除包含 NaN 的行
df.dropna(axis=1)              # 删除包含 NaN 的列
df.dropna(thresh=3)            # 至少有 3 个非空值才保留
df.dropna(subset=['age'])      # 只检查 age 列的缺失值

# 填充
df.fillna(0)                   # 用 0 填充所有缺失值
df.fillna({'age': 0, 'name': '未知'})  # 不同列用不同值
df.fillna(method='ffill')      # 用前一行的值填充（向前填充）
df.fillna(method='bfill')      # 用后一行的值填充（向后填充）
df['age'].fillna(df['age'].median())  # 用中位数填充
```

## 2. 重复值处理

```python
df.duplicated()              # 检测重复行，返回布尔 Series
df.duplicated(subset=['name'])  # 只根据 name 列判断重复
df.drop_duplicates()         # 删除重复行（保留第一次出现）
df.drop_duplicates(keep='last')   # 保留最后一次出现
df.drop_duplicates(subset=['name', 'age'])  # 指定列去重
```

## 3. 数据类型转换

```python
df['age'] = df['age'].astype(int)              # 转为整数
df['price'] = df['price'].astype('float64')    # 转为浮点
pd.to_numeric(df['col'], errors='coerce')       # 强制转数值，无效变 NaN
pd.to_datetime(df['date'])                      # 转为日期时间
df['category'] = df['category'].astype('category')  # 转为分类类型（省内存）
```

## 4. 字符串处理

```python
# 通过 .str 访问器操作 Series 中的字符串
df['name'].str.upper()        # 转大写
df['name'].str.lower()        # 转小写
df['name'].str.strip()        # 去除首尾空格
df['name'].str.replace('_', ' ')  # 替换
df['name'].str.contains('张')    # 是否包含
df['name'].str.split('_')        # 分割
df['name'].str.len()             # 字符串长度
```

# 七、数据变换

## 1. 增删列

```python
# 增加列
df['new_col'] = df['age'] * 2                    # 直接赋值
df.insert(1, 'new_col', values)                   # 在指定位置插入列
df['full_name'] = df['first'] + ' ' + df['last']  # 字符串拼接

# 删除列
df.drop('col_name', axis=1, inplace=True)          # 删除单列
df.drop(['col1', 'col2'], axis=1)                  # 删除多列
df.drop(columns=['col1', 'col2'])                  # 更直观的写法

# 删除行
df.drop(0, axis=0)                                 # 按索引删除行
df.drop([0, 1, 2])                                 # 删除多行
```

## 2. 重命名

```python
df.rename(columns={'旧名': '新名', 'age': '年龄'})
df.rename(index={0: 'first', 1: 'second'})
df.columns = ['col1', 'col2', 'col3']  # 批量重命名
```

## 3. 排序

```python
df.sort_values('age')                    # 按 age 升序排列
df.sort_values('age', ascending=False)   # 降序
df.sort_values(['city', 'age'], ascending=[True, False])  # 多列排序
df.sort_index()                          # 按索引排序
```

## 4. apply 与 map

```python
# apply：对行或列应用函数
df['age'].apply(lambda x: x * 2)                # 对 Series 逐元素
df.apply(lambda row: row['a'] + row['b'], axis=1)  # 对行操作
df.apply(np.mean, axis=0)                       # 对每列求均值

# map：Series 元素映射替换
df['gender'].map({'M': '男', 'F': '女'})

# applymap：DataFrame 逐元素应用函数
df.applymap(lambda x: round(x, 2))
```

| 方法 | 作用对象 | 说明 |
|------|---------|------|
| `map` | Series | 元素替换映射 |
| `apply` | Series / DataFrame | 沿轴应用函数 |
| `applymap` | DataFrame | 逐元素应用函数 |

# 八、分组聚合

这是 Pandas 最强大的功能之一，类似 SQL 的 `GROUP BY`：

```python
# 基本分组聚合
df.groupby('city')['salary'].mean()   # 按城市分组，求平均薪资

# 多列聚合
df.groupby('city').agg({
    'salary': 'mean',
    'age': ['min', 'max', 'mean']
})

# 常用聚合函数
df.groupby('department').agg(
    人数=('name', 'count'),
    平均薪资=('salary', 'mean'),
    最高薪资=('salary', 'max'),
    最低薪资=('salary', 'min'),
    薪资总和=('salary', 'sum'),
    薪资标准差=('salary', 'std'),
)

# 多级分组
df.groupby(['city', 'gender'])['salary'].mean()

# transform：保持原维度
df['avg_by_city'] = df.groupby('city')['salary'].transform('mean')
# 每个人获得自己所在城市的平均薪资

# filter：按分组条件筛选
df.groupby('city').filter(lambda g: len(g) >= 3)
# 只保留成员数 ≥ 3 的城市
```

# 九、合并与连接

| 方法 | 说明 | 类比 SQL |
|------|------|---------|
| `pd.concat([df1, df2])` | 纵向/横向拼接 | UNION |
| `df.merge(df2, on='key')` | 基于键的列连接 | JOIN |
| `df.join(df2, on='key')` | 基于索引的连接 | JOIN |

```python
# concat：纵向堆叠
pd.concat([df1, df2], axis=0, ignore_index=True)     # 纵向（追加行）
pd.concat([df1, df2], axis=1)                        # 横向（追加列）

# merge：类似 SQL JOIN
df1.merge(df2, on='id')                               # 内连接（默认）
df1.merge(df2, on='id', how='left')                   # 左连接
df1.merge(df2, on='id', how='right')                  # 右连接
df1.merge(df2, on='id', how='outer')                  # 全连接
df1.merge(df2, left_on='user_id', right_on='id')      # 键名不同时
df1.merge(df2, on=['col1', 'col2'])                   # 多键连接

# join：按索引连接
df1.join(df2, how='inner')
```

⚠️ **merge vs concat**：merge 是按列值匹配连接（水平合并），concat 是简单堆叠（竖直或水平拼接）。

# 十、时间序列

```python
# 创建时间序列
dates = pd.date_range('2024-01-01', periods=100, freq='D')  # 100 天
dates = pd.date_range('2024-01-01', '2024-12-31', freq='W') # 每周

# 设为时间索引
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 按时间筛选
df.loc['2024-01']          # 2024 年 1 月的数据
df.loc['2024-01':'2024-03']  # 1 到 3 月的数据
df.loc['2024-01-15':'2024-01-20']

# 重采样（resample）
df.resample('M')['sales'].sum()      # 按月汇总 -> 月销售额
df.resample('W')['sales'].mean()     # 按周平均 -> 周均销售额
df.resample('Q')['sales'].sum()      # 按季度汇总

# 常用频率
# 'D': 天, 'W': 周, 'M': 月, 'Q': 季度, 'Y': 年
# 'H': 小时, 'T'/'min': 分钟, 'S': 秒

# 时间特征提取
df['year']   = df.index.year
df['month']  = df.index.month
df['day']    = df.index.day
df['weekday'] = df.index.weekday   # 0=周一, 6=周日
```

# 十一、数据透视表

```python
# pivot_table：类似 Excel 数据透视表
pd.pivot_table(df, 
    values='sales',      # 值
    index='city',        # 行
    columns='year',      # 列
    aggfunc='sum',       # 聚合函数
    fill_value=0,        # 空值填充
    margins=True         # 是否显示汇总
)

# pivot：数据重塑（不聚合）
df.pivot(index='date', columns='city', values='temperature')

# melt：宽表转长表
pd.melt(df, 
    id_vars=['name'],       # 保持不变的列
    value_vars=['math', 'english'],  # 要融合的列
    var_name='subject',     # 新列名（原列名放入此列）
    value_name='score'      # 新列名（原值放入此列）
)
```

# 十二、常用统计方法

```python
# 基本统计
df['col'].mean()      # 均值
df['col'].median()    # 中位数
df['col'].mode()      # 众数
df['col'].std()       # 标准差
df['col'].var()       # 方差
df['col'].quantile(0.25)  # 四分位数
df['col'].value_counts()  # 频数统计
df['col'].corr(df['col2'])  # 相关系数

# 全表统计
df.corr()             # 数值列相关系数矩阵
df.cov()              # 协方差矩阵
df['col'].cumsum()    # 累计和
df['col'].pct_change()  # 环比变化率

# 排名
df['col'].rank(ascending=False)
df['col'].nlargest(10)  # 最大的 10 个值
df['col'].nsmallest(10)  # 最小的 10 个值
```

# 十三、常用技巧速查

| 操作 | 代码 |
|------|------|
| 查看所有列名 | `df.columns.tolist()` |
| 列名去空格 | `df.columns = df.columns.str.strip()` |
| 重置索引 | `df.reset_index(drop=True)` |
| 设置索引 | `df.set_index('id')` |
| 随机抽样 | `df.sample(n=100)` 或 `df.sample(frac=0.1)` |
| 条件替换 | `df['col'].where(df['col'] > 0, 0)` 或 `np.where` |
| 按条件分组标记 | `pd.cut(df['age'], bins=[0,18,60,100], labels=['未成年','成年','老年'])` |
| 内存优化 | `df.info(memory_usage='deep')` 检查内存，转 category 类型 |
| 查看大 DataFrame | `pd.set_option('display.max_columns', None)` 显示全部列 |
| 链式操作 | 用括号包裹多行：`(df.query(...).groupby(...).agg(...))` |

# 十四、数据可视化（快速预览）

```python
import matplotlib.pyplot as plt

# Pandas 内置绘图（基于 Matplotlib）
df['col'].plot(kind='line')      # 折线图
df['col'].plot(kind='bar')       # 柱状图
df['col'].plot(kind='hist')      # 直方图
df['col'].plot(kind='box')       # 箱线图
df.plot(kind='scatter', x='col1', y='col2')  # 散点图
df.corr().style.background_gradient(cmap='coolwarm')  # 相关性热力图

plt.show()
```

# 十五、性能优化建议

1. **优先使用向量化操作**，避免逐行 `iterrows()`
2. **用 `apply()` 代替显式循环**，但仍不如向量化快
3. **选择合适的数据类型**：`category` 比 `object` 省内存，数值类型比 `object` 快得多
4. **批量读写**：用 `chunksize` 分块读取大文件
5. **用 `query()` 和 `eval()`** 加速复杂条件筛选

```python
# ❌ 慢：逐行遍历
for idx, row in df.iterrows():
    df.at[idx, 'new'] = row['a'] + row['b']

# ✅ 快：向量化
df['new'] = df['a'] + df['b']
```

# 十六、常见坑与注意事项

- **SettingWithCopyWarning**：链式索引赋值时容易出现，用 `.loc[]` 一步完成赋值
  ```python
  # ❌ 可能出错
  df[df['age'] > 30]['name'] = 'old'
  
  # ✅ 正确写法
  df.loc[df['age'] > 30, 'name'] = 'old'
  ```

- **`inplace` 参数**：多数操作默认返回新对象，需赋值或设 `inplace=True`
- **`axis=0` vs `axis=1`**：0 是沿行方向（操作列），1 是沿列方向（操作行）
- **NaN 不等于任何值，包括它自己**：用 `pd.isna()` 检测，不要用 `== np.nan`
- **merge 时注意重复键**：可能产生笛卡尔积

---

> **学习路径建议**：先掌握创建、读写、`loc/iloc` 索引、缺失值处理、groupby 聚合、merge 连接这六个核心模块，就能应对 80% 的数据分析场景。其余遇到具体需求再查阅即可。
