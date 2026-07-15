---
tags:
  - Python
related:
  - "[[深度学习基础知识点]]"
---

## 1. class —— 定义类

`class` 是创建对象的模板。方法就是定义在类里面的函数。

定义类后面的括号内是==父类（继承）==
告诉 Python：这个类的**爸爸是谁**，继承父类的所有方法和属性。
```python
class Dog(Animal):          # Dog 继承 Animal
    pass

class SAMPLE_Dataset(Dataset):  # SAMPLE_Dataset 继承 Dataset
    pass
```

没括号的情况
```python
class SimpleClass:        # 没写括号 = 隐式继承 object
    pass
```
所有自定义类最终都追溯到 `object`。

### 抽象基类（Abstract Base Class，ABC）
**抽象基类 = 只定义接口，不实现（或只实现通用部分），子类必须填空。**
相关的叫法：

| 名称                    | 含义                                              |
| --------------------- | ----------------------------------------------- |
| **抽象基类 / ABC**        | 包含未实现方法的类，不能直接实例化                               |
| **接口 / Interface**    | 只规定方法签名，完全没有实现（Python 没有原生 interface，Java/C# 有） |
| **父类 / Parent Class** | 通用的叫法，不限于抽象还是具体                                 |
| **超类 / Superclass**   | 同父类，更学术的说法                                      |

例如`Dataset` 是 PyTorch 定义好的**抽象基类**，它规定了一套"接口契约"：
```python
# Dataset 的契约
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError   # 子类必须覆盖

    def __len__(self):
        raise NotImplementedError   # 子类必须覆盖

    def __add__(self, other):       # 已实现，支持 dataset1 + dataset2 拼接
        return ConcatDataset([self, other])

```

```python
# 你只需要实现两个方法，剩下的 PyTorch 全包了：
class SAMPLE_Dataset(Dataset):
    def __len__(self):          # ← 必须实现：告诉 DataLoader 有多少样本
        return len(self.sample_path_label)

    def __getitem__(self, idx): # ← 必须实现：告诉 DataLoader 第 idx 个样本是什么
        # 读图、转tensor、返回
        return sample
```
PyTorch 拿到这两个后替你做的事:
- `DataLoader` 在循环时自动 `for i in range(len(dataset))` → `dataset[i]`
- 多线程加载：每个 worker 各持一份 dataset，各自调 `__getitem__`
- 可以 `dataset_a + dataset_b` 拼接两个数据集（`__add__` 已在父类实现）
- 可以用 `Subset(dataset, [0,1,5])` 切片

本质上就是模板方法模式：框架定好骨架，你填两个方法即可。 这就是面向对象里"面向接口编程"的体现。

PyTorch 的 `Dataset` 实际上没用到 `abc` 模块，而是用了"鸭子类型"——只要你实现了 `__len__` 和 `__getitem__`，不管是否显式继承 `Dataset`，DataLoader 都能用。但继承它是一种**声明意图**，让代码更清晰。

## 2. 调用类里面的方法

类里的函数叫"方法"，调用方式取决于它的类型。

### 2.1 普通方法：先创建实例，再调

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(f"{self.name}：汪汪")

dog = Dog("旺财")    # ① 创建实例
dog.bark()           # ② 实例.方法()  →  旺财：汪汪
```

### 2.2 静态方法：直接用类名调

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

Math.add(1, 2)   # 类名.方法()，不用创建实例
```

### 2.3 DDPM 中：`model(t)` 怎么就能调 `forward`？

PyTorch 的 `nn.Module` 实现了 `__call__`，让你可以像调函数一样调实例：

```python
model = SinusoidalPosEmb(dim=128)   # 创建实例
emb = model(t)                      # 等价于 model.forward(t)
```

普通 Python 类也能这样：

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):          # 实现 __call__
        return x + self.n

add5 = Adder(5)
add5(10)   # 15  —— 实例可以当函数用
```

### 速查

| 方法类型 | 定义 | 调用方式 | 需要实例？ |
|---------|------|---------|:---:|
| 普通方法 | `def foo(self, x)` | `obj.foo(x)` | ✅ |
| 静态方法 | `@staticmethod` + `def foo(x)` | `Cls.foo(x)` | ❌ |
| `__call__` | `def __call__(self, x)` | `obj(x)` | ✅ |

## 3. `__init__` —— 构造函数

创建对象时**自动调用**，用来设置初始状态。

```python
class Dog:
    def __init__(self, name, age):
        self.name = name    # 把参数存到实例上
        self.age = age

dog = Dog("旺财", 3)        # 自动调用 __init__
print(dog.name)             # 旺财
print(dog.age)              # 3
```

**不是** `dog.__init__("旺财", 3)`，而是 `Dog("旺财", 3)`，Python 自动把 `__init__` 转成构造调用。

## 4. `self` —— 指代"当前这个实例"

`self` 不是关键字，是**约定俗成的第一个参数名**。它指向调用方法的那个具体对象。

```python
class Dog:
    def __init__(self, name):
        self.name = name        # self.name 属于这个实例

    def bark(self):
        print(f"{self.name}：汪汪")   # 这里的 self 是调用时的那个狗

dog1 = Dog("旺财")
dog2 = Dog("来福")
dog1.bark()   # self = dog1 → "旺财：汪汪"
dog2.bark()   # self = dog2 → "来福：汪汪"
```

**一句话**：`self` = 谁调用这个方法，它就指谁。

## 5. `self.xxx` —— 实例属性

`self.xxx = 值` 把数据挂在实例身上，之后整个类的任何方法里都能通过 `self.xxx` 访问。

```python
class Counter:
    def __init__(self):
        self.count = 0          # 初始化实例属性

    def add(self, n):
        self.count += n         # 读写实例属性

    def show(self):
        print(self.count)       # 读取实例属性

c = Counter()
c.add(5)
c.add(3)
c.show()   # 8
```

## 6. `super().__init__()` —— 调用父类初始化

当你的类继承别人的类时，自己的 `__init__` 会**覆盖**掉父类的。必须手动调用 `super().__init__()` 让父类也完成初始化。

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)    # 让 Animal 设置 self.name
        self.breed = breed        # 再设置自己的

dog = Dog("旺财", "金毛")
print(dog.name)    # 旺财     ← 来自父类
print(dog.breed)   # 金毛     ← 来自自己
```

**不写 `super().__init__()` 的后果**：父类的属性不会被设置，访问会报错。

## 7. 实战对照：DDPM 中的类

```python
class SinusoidalPosEmb(nn.Module):          # 继承 nn.Module
    def __init__(self, dim: int):           # 构造时传入 dim
        super().__init__()                  # ① 让 nn.Module 初始化（必须）
        self.dim = dim                      # ② 把 dim 存为实例属性

    def forward(self, t):                   # self = 调用时的那个模型实例
        half_dim = self.dim // 2            # ③ 通过 self 读取 dim
        ...
        return emb
```

一步步拆解：

```python
model = SinusoidalPosEmb(dim=128)   # → 自动调用 __init__(self=model, dim=128)
                                    #   → super().__init__() 初始化 nn.Module 部分
                                    #   → self.dim = 128

emb = model(t)                      # → 自动调用 forward(self=model, t)
                                    #   → 内部 self.dim 就是 128
```

| 概念 | 你看到 | 实际发生了什么 |
|------|--------|---------------|
| `__init__(self, dim)` | 定义时有两个参数 | 调用时只传 `dim`，`self` 自动传入 |
| `self.dim = dim` | 赋值 | 把值存到**这个**模型实例上 |
| `super().__init__()` | 一行调用 | 让父类 `nn.Module` 完成初始化 |
| `self.dim // 2` | 读取属性 | 从实例上取出之前存的 `dim` |

## 8. 装饰器 —— `@xxx` 的本质

### 8.1 一句话理解

装饰器就是一个**包裹函数的函数**：你把函数 A 扔进去，它返回一个增强了的新函数。
`@` 只是语法糖，本质是**函数调用**。

### 8.2 无参数装饰器示例代码
#### 无参数 + 不修改返回值（纯增强功能）
最标准的日志/计时器，只加副作用，原样返回结果。

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)  # 执行原函数
        cost = time.perf_counter() - start
        print(f"[耗时] {func.__name__}: {cost:.3f}s")
        return result  # 原封不动返回
    return wrapper

@timer
def add(x, y):
    time.sleep(0.5)
    return x + y

print(add(3, 5))  # 输出耗时信息，然后打印 8（返回值没变）
```

#### 无参数 + 修改返回值（转换输出）
比如把返回的字符串**自动大写**，或者把数字**翻倍**。
```python
import functools

def double_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * 2  # 修改！把计算结果翻倍
    return wrapper

@double_result
def get_number():
    return 21

print(get_number())  # 输出 42（原函数返回21，被装饰器乘2了）
```



### 8.3 带参数的装饰器

有些装饰器自己也需要参数（如 `@lru_cache(maxsize=128)`）。这时需要**再包一层**。带参数的装饰器必须**三层嵌套**。
#### 有参数 + 不修改返回值（控制过程行为）
最经典的例子是**带重试机制的装饰器**：参数控制重试几次，但只要原函数最终成功，返回值必须原封不动地交给调用者

```python
import functools
import time
import random

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    # 执行原函数，拿到结果
                    result = func(*args, **kwargs)
                    # 注意这里：只要不报错，直接 return 原结果，绝对不改
                    return result  
                except Exception as e:
                    last_exception = e
                    print(f"[重试] 第 {attempt} 次失败，{delay}s 后重试...")
                    time.sleep(delay)
            # 全部重试失败，抛出异常（注意，这里抛异常，相当于也没改返回值）
            raise last_exception
        return wrapper
    return decorator

# 使用：配置最多重试5次，每次间隔2秒
@retry(max_attempts=5, delay=2)
def unstable_api_call():
    if random.random() < 0.7:  # 70%概率失败
        raise ConnectionError("网络超时")
    return "成功数据"

# 调用者拿到的永远是原函数返回的 "成功数据"，装饰器只加了重试过程
data = unstable_api_call()  
print(data)  # 输出: 成功数据（返回值没变）
```

#### 有参数 + 修改返回值（带配置的转换）
给返回值**乘以指定倍数**，或加上指定**前后缀**。
```python
import functools

# 注意这是三层函数
def multiply(factor):          # 第一层：接收装饰器参数
    def decorator(func):       # 第二层：接收原函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # 第三层：接收原函数调用参数
            result = func(*args, **kwargs)
            return result * factor     # 修改！使用外部传入的 factor
        return wrapper
    return decorator

@multiply(factor=10)  # 带括号调用，传入参数
def get_score():
    return 8.5

print(get_score())  # 输出 85.0（原函数返回8.5，被装饰器乘了10）
```

### 8.4 常见内置装饰器速查

| 装饰器 | 作用 | 示例 |
|--------|------|------|
| `@staticmethod` | 方法不需要 `self` | `def add(a, b):` 无 self |
| `@classmethod` | 第一个参数是类 `cls` | `def from_str(cls, s):` |
| `@property` | 方法变属性，不加括号调用 | `obj.name` 而非 `obj.name()` |
| `@functools.lru_cache` | 缓存函数返回值 | `@lru_cache(maxsize=128)` |
| `@dataclass` | 自动生成 `__init__` | `class Point: x: int; y: int` |



## 9. `__Func__`——魔术方法
`__Func__`（双下划线包裹）这样的函数/方法，统称为**特殊方法（Special Methods）**，更通俗的叫法是**魔术方法（Magic Methods）**

它们不是让你直接调用的普通函数，而是**Python 内部协议的钩子**。当你对对象使用特定的语法或内置函数时，Python 解释器会自动去调用对应的特殊方法。

### 1. 核心特征（怎么用）

- **无需手动调用**：你几乎不会写 `obj.__add__(other)`，而是写 `obj + other`。
- **由解释器触发**：`len(obj)` 触发 `obj.__len__`，`print(obj)` 触发 `obj.__str__`。
- **定义在类内部**：用于让自定义类的实例表现得像内置类型（如列表、字符串、数字）一样。

### 2. 常见分类

|类别|常见方法|触发时机|
|---|---|---|
|**对象生命周期**|`__new__`, `__init__`|创建和初始化实例（`Class()`）|
|**销毁**|`__del__`|对象被垃圾回收时|
|**字符串/字节表示**|`__str__`, `__repr__`|`print()` 或 `str()`；交互式环境直接显示|
|**算术运算符**|`__add__` (+), `__sub__` (-), `__mul__` (*)|对象进行数学运算时|
|**比较运算符**|`__eq__` (==), `__lt__` (<), `__gt__` (>)|对象比较大小或排序时|
|**容器/序列协议**|`__len__`, `__getitem__` (索引), `__setitem__` (赋值)|`len()`、`obj[key]`、`for` 循环迭代|
|**可调用对象**|`__call__`|把实例当作函数调用（`obj()`）|
|**上下文管理**|`__enter__`, `__exit__`|`with` 语句块|
|**属性访问**|`__getattr__`, `__setattr__`|获取或设置不存在的属性时|

### 3. 代码演示
```python
class Vector:
    def __init__(self, x, y):          # 构造
        self.x = x
        self.y = y

    def __add__(self, other):          # 重载 + 号
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):                 # 重载 print()
        return f"Vector({self.x}, {self.y})"

    def __len__(self):                 # 重载 len()
        return 2

    def __call__(self, scale):         # 让实例可调用
        return Vector(self.x * scale, self.y * scale)

# 使用演示
v1 = Vector(1, 2)
v2 = Vector(3, 4)

v3 = v1 + v2          # 自动触发 __add__
print(v3)             # 自动触发 __str__  -> 输出: Vector(4, 6)
print(len(v3))        # 自动触发 __len__  -> 输出: 2

v4 = v3(2)            # 自动触发 __call__ -> 相当于把向量放大2倍
print(v4)             # 输出: Vector(8, 12)
```



# 10. enumerate()
`enumerate()` 是 Python 的内置函数，它将一个可迭代对象（如列表、字符串、元组等）包装成一个**枚举对象**，在迭代时同时产出 `(索引, 元素)` 对。
```python
enumerate(iterable, start=0)
# iterable：可迭代对象（必填）
# start：索引起始值，默认为 `0`（可选）
```

❌ **没有 enumerate 的写法**（手动维护计数器）：
```python
fruits = ["苹果", "香蕉", "橙子"]

i = 0
for fruit in fruits:
    print(i, fruit)
    i += 1
```

✅ **使用 enumerate**（简洁且 Pythonic）：
```python
fruits = ["苹果", "香蕉", "橙子"]

for i, fruit in enumerate(fruits):  # for后的变量名可随便取，个数必须匹配
    print(i, fruit)

# 输出:
# 0 苹果
# 1 香蕉
# 2 橙子
```

```python
# ❌ 错误：只用了一个变量，拿到的是整个元组
for x in enumerate(fruits):
    print(x)           # (0, '苹果')  → 不是你想要的效果
```


# 11. 可迭代对象

**可迭代对象**就是**可以被 `for` 循环遍历的东西**。
凡是满足下面这个条件的，就是可迭代对象：
```python
for x in obj:   # 如果这行能跑不报错
    ...         # 那 obj 就是可迭代对象
```

## 你每天都在用

|可迭代对象|例子|for 出来的|
|---|---|---|
|列表|`[1, 2, 3]`|`1`, `2`, `3`|
|字符串|`"abc"`|`"a"`, `"b"`, `"c"`|
|range|`range(5)`|`0`, `1`, `2`, `3`, `4`|
|字典|`{"a": 1, "b": 2}`|`"a"`, `"b"`（键）|
|文件对象|`open("a.txt")`|一行一行的文本|
|DataLoader|`train_dataloader`|一批一批的 `(images, labels)`|
|Dataset|`training_data`|一个一个的 `(image, label)`|

## 它和迭代器的关系

| |可迭代对象|迭代器|
|---|---|---|
|是什么|一个**仓库**，里面有东西|一个**指针**，指着当前拿到哪了|
|能直接 `next()` 吗？|❌|✅|
|怎么变成迭代器|`iter(可迭代对象)` → 迭代器|—|
|有状态吗？|无（取不完）|有（取一次少一个，取完就没了）|

```python
# 列表是可迭代对象，但没状态
lst = [1, 2, 3]
next(lst)          # ❌ 报错！列表不知道"取到哪了"

# 给它配个指针（迭代器），就有状态了
it = iter(lst)      # 迭代器，指着第一个元素之前
next(it)            # 1
next(it)            # 2
next(it)            # 3
next(it)            # ❌ StopIteration，取完了
```

## `for` 循环帮你干了什么

```python
for x in [1, 2, 3]:
    print(x)
```
等价于 Python 在幕后：
```python
it = iter([1, 2, 3])    # ① 把可迭代对象变成迭代器
while True:
    try:
        x = next(it)    # ② 一次次往后取
        print(x)
    except StopIteration:
        break           # ③ 取完了就停
```
所以你写 `for` 的时候，`iter()` 和 `next()` 一个都没少，只是你看不到。

## 一句话总结

> **可迭代对象** = 能放进 `for ... in ...` 的任何东西。它内部实现了 `__iter__` 方法，告诉 Python "我知道怎么创建一个迭代器来遍历自己"。
