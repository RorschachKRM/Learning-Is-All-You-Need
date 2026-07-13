---
title: 数据集类、sample_dataset.py
tags:
  - DDPM
  - Dataset
related:
---
# 概念
==功能一句话解释：**把硬盘上的 SAR 图片变成 PyTorch 模型能吃的训练数据。**==

训练 DDPM 时，你需要不断喂给它：
- 一张 SAR 图像（128×128 灰度图）
- 它的类别标签（"这是一辆坦克"还是"一辆装甲车"）
- 它的方位角（"这张图是从 60° 角度拍的"）
这个类就是负责**自动化地把硬盘上的图片读出来，打包好这些信息，交给模型**。

`SAMPLE_Dataset` 类本身是通用的，不管是 DDPM、SNGAN 还是其他生成模型，数据加载部分完全一样。
 
## 核心问题：数据在哪里？长什么样？
硬盘上的目录结构是这样的：
```
dataset/SAMPLE/png_images/qpm/real/
├── 2s1/          ← 全是 2S1 自行火炮的 SAR 图像
│   ├── 2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
│   ├── 2s1_real_A_elevDeg_016_azCenter_090_50_serial_b02.png
│   └── ...
├── bmp2/         ← 全是 BMP2 步兵战车的 SAR 图像
├── btr70/
├── m1/
├── m2/
├── m35/
├── m60/
├── m548/
├── t72/
└── zsu23/
```

两个关键信息隐藏在**文件名**和**目录名**里：
```
2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
                                    ^^^^^ ^^
                                    60.22° ← 这张图是从 60.22° 方位角拍摄的
↑
'2s1' 目录 → 类别标签 = 0
```
这个文件做的事情就是**自动提取这些信息，而不是让用户手动标注**。

## 具体实现：
### 1. `SAMPLE_Dataset` 类 — 把数据变成标准接口

它继承 PyTorch 的 `Dataset`，实现了三个必须的方法：
```
DataLoader 调用流程：
                      
  DataLoader 问: "你有几个样本？"
         │
         ▼
  __len__()  →  return 2747    ← 返回总数
         │
         │  DataLoader 循环: for i in range(2747)
         ▼
  __getitem__(i)  →  返回第 i 个样本的字典:
                      {
                        'image': Tensor(1,128,128),  ← 模型输入
                        'label': 3,                   ← 类别标签
                        'az': 60.22,                  ← 方位角度数
                        'name': '2s1_real_...'        ← 文件名
                      }
```

**`__init__` 做了什么？** 扫描数据源，生成一个列表 `[[路径1, 标签1], [路径2, 标签2], ...]`。这是唯一一次访问硬盘，之后 `__getitem__` 按索引读就行。

**`__getitem__` 做了什么？** 从列表取出一条 → 打开图片 → 灰度化 → 归一化到 [0,1] → 转成 tensor → 从文件名提取方位角 → 打包返回。

**为什么要有两种加载方式（txt_file vs data_root）？** 因为`data_root` 扫描目录很方便但慢（每次启动要遍历所有文件），`txt_file` 需要提前准备但快（直接读文件列表）。实际使用流程是：先用 `data_root` 生成 txt，之后训练都用 txt。

### 2. `create_sample_txt_file` 函数 — 生成 txt 文件
```python
data_root = "dataset/SAMPLE/png_images/qpm/real"

create_sample_txt_file(data_root, "data/sample_train.txt")
```
运行一次，生成 `sample_train.txt`：
```
real/2s1/2s1_real_A_...azCenter_060_22_...png 0
real/bmp2/bmp2_real_A_...azCenter_035_49_...png 1
real/btr70/btr70_real_A_...azCenter_120_10_...png 2
...
```
之后训练脚本直接用 `txt_file="data/sample_train.txt"` 加载，不再扫描目录。





# 代码1
（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
## `__init__`初始化函数
`__init__` 做的事情就是：**统一数据加载入口 → 构建一个 `[路径, 标签]` 列表**。
后续 `__getitem__` 只需按索引从这个列表取数据，不用关心数据来源是 txt 还是目录扫描。

```python
class SAMPLE_Dataset(Dataset):
    """
    SAMPLE dataset for DDPM training
    """

    def __init__(self, data_root=None, txt_file=None, transform=None):
        """
        参数：
			data_root：数据集根目录路径（如果使用 txt_file，则可选）
			txt_file：txt 文件路径(path, label)（如果使用 data_root，则可选）
			transform：要应用于样本的可选转换
        """
```

| 参数          | 类型         | 默认值    | 作用                                 |
| ----------- | ---------- | ------ | ---------------------------------- |
| `data_root` | `str`      | `None` | 数据集根目录路径，内含按类别分好的子文件夹              |
| `txt_file`  | `str`      | `None` | 预先生成的 txt 文件路径，每行记录图片路径和标签         |
| `transform` | `callable` | `None` | 图像预处理函数（如 `transforms.ToTensor()`） |
### 有关Dataset
`Dataset` 是从第3行导入的：
```python
from torch.utils.data import Dataset
```
它是 **PyTorch 的抽象基类** `torch.utils.data.Dataset`，不是参数，而是 `SAMPLE_Dataset` 继承的父类。
在 Python 里，`class SAMPLE_Dataset(Dataset)` 的括号里是**父类列表**，表示 `SAMPLE_Dataset` 继承自 `Dataset`。
PyTorch 要求==自定义数据集==必须继承这个类并实现两个方法：
- `__len__` → 你的第96行
- `__getitem__` → 你的第99行
这样 `SAMPLE_Dataset` 就能被 `torch.utils.data.DataLoader` 直接使用。

### 有关DataLoader
`DataLoader` 是 PyTorch 的数据加载器，来自 `torch.utils.data.DataLoader`。
Dataset 管"怎么读一个样本"，DataLoader 管"怎么攒一批样本"。
核心作用：把 `Dataset`（你的 `SAMPLE_Dataset`）包装成**可批量迭代**的对象，自动帮你做：

| 功能        | 说明                                                  |
| --------- | --------------------------------------------------- |
| **批量打包**  | 把多个样本堆叠成 batch tensor                               |
| **随机打乱**  | `shuffle=True` 每 epoch 洗牌                           |
| **多线程加载** | `num_workers` 并行读数据，不阻塞 GPU                         |
| **自动批处理** | 通过 `collate_fn` 把 list of dict 合并成 batch of tensors |


### transform属性：有关数据增强/预处理函数
```python
		self.transform = transform
```
把数据增强/预处理函数存为实例属性，供后面 `__getitem__` 使用。如果传了 `None`，`__getitem__` 里会走默认逻辑（手动转 tensor）。


### classes属性
```python
		self.classes = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
```
这是 MSTAR SAR 数据集的标准 10 类军事目标。列表的**索引天然对应标签编号**：
'2s1'→0, 'bmp2'→1, ..., 'zsu23'→9


### class_to_idx属性
```python
		self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
```

字典推导式一行顶一个 for 循环。**推导式通用格式：**
```python
{key_expr(键): value_expr(值) for 变量 in 可迭代对象}
```

`enumerate()` 把列表变成 `(索引, 元素)` 的迭代器：

| 迭代   | idx | cls_name  |
| ---- | --- | --------- |
| 第1轮  | 0   | `'2s1'`   |
| 第2轮  | 1   | `'bmp2'`  |
| 第3轮  | 2   | `'btr70'` |
| ...  | ... | ...       |
| 第10轮 | 9   | `'zsu23'` |

最终得到：{
    '2s1': 0,
    'bmp2': 1,
    'btr70': 2,
    'm1': 3,
    'm2': 4,
    'm35': 5,
    'm60': 6,
    'm548': 7,
    't72': 8,
    'zsu23': 9
}
**为什么需要这个？** `scan_dataset_directory` 遍历子目录时，目录名是字符串（如 `"bmp2"`），需要快速查到对应的数字标签 `1`。字典查找是 O(1)。


### sample_path_label属性：`[图片路径, 标签]`列表
```python
		if txt_file:                                          # 分支 A
		    self.sample_path_label = self.read_dataset_txt(txt_file)
		elif data_root:                                       # 分支 B
		    self.sample_path_label = self.scan_dataset_directory(data_root)
		else:                                                 # 兜底
		    raise ValueError("Either data_root or txt_file must be provided")
```
两个分支最终都产生同一种数据结构 `self.sample_path_label`，它是一个列表：
```
[
    ["real/2s1/xxx.png", 0],    # [图片路径(str), 标签(int)]
    ["real/bmp2/yyy.png", 1],
    ...
]
```

|          | 分支 A：`txt_file`      | 分支 B：`data_root`           |
| -------- | -------------------- | -------------------------- |
| **触发条件** | 传了 txt 文件路径          | 传了根目录路径                    |
| **调用方法** | `read_dataset_txt()` | `scan_dataset_directory()` |
| **标签来源** | txt 文件里直接写好的数字       | 从子目录名 + `class_to_idx` 推断  |
| **优点**   | 速度快，数据划分明确           | 无需预处理，即开即用                 |
如果有 txt 又有 data_root，优先走 txt（`if-elif` 结构决定的）。
两个都不传就抛异常，因为没有任何数据来源。



## init方法1：从txt文件读取解析
```python
    def read_dataset_txt(self, txt_file):
        """Read dataset from txt file"""
        list_path_label = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    file_path = parts[0]
                    label = int(parts[1])
                    list_path_label.append([file_path, label])
        return list_path_label
```
所需的**txt文件格式**：
```
real/2s1/2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png 0
real/bmp2/bmp2_real_A_elevDeg_016_azCenter_035_49_serial_9563.png 1
...
```
每行：`图片路径 类别编号`

## init方法2：从目录扫描解析
```python
def scan_dataset_directory(self, data_root):
        """Scan dataset directory and extract all images with labels and azimuth angles"""
        list_path_label = []  # 空列表，用来收集结果。每个元素是 `[文件路径, 数字标签]`。
        for class_name in self.classes:  # 按固定顺序遍历10个类别，保证标签一致性
            class_dir = os.path.join(data_root, class_name)  # 拼接路径
            
            if not os.path.exists(class_dir):  # 如果某个类别文件夹不存在，打印警告并跳过。这允许数据集不包含全部10类（比如只用部分类别训练）
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
                
            for filename in os.listdir(class_dir): # 列出该类别文件夹下的所有文件名
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')): #  过滤，只保留图片文件
                    file_path = os.path.join(class_dir, filename) # 图片完整路径
                    
                    label = self.class_to_idx[class_name]  # 通过之前建的字典查表，提取标签
                    list_path_label_az.append([file_path, label])  # 追加一条[路径, 标签]到结果列表
                    
        return list_path_label
```
当用户传入 `data_root` 而非 `txt_file` 时调用。**从目录结构推断标签**，无需预先准备 txt 文件。预期的目录结构：
```
data_root/
├── 2s1/        → label = 0
│   ├── 2s1_real_A_..._azCenter_060_22_...png
│   └── ...
├── bmp2/       → label = 1
├── btr70/      → label = 2
├── m1/         → label = 3
├── m2/         → label = 4
├── m35/        → label = 5
├── m60/        → label = 6
├── m548/       → label = 7
├── t72/        → label = 8
└── zsu23/      → label = 9
```
扫描过程：遍历 `self.classes` 中的每个类别名 → 进入对应子目录 → 收集所有图片文件及其完整路径 → 标签由目录名经 `class_to_idx` 映射得到。

方式对比：

|方式|优点|缺点|
|---|---|---|
|`txt_file`|加载快，可灵活划分训练/测试集|需预先生成 txt 文件|
|`data_root`|即开即用，无需预处理|不能控制数据划分|


## 方位角提取
```python
    def extract_azimuth_from_filename(self, filename):
        """
        从文件名中提取方位角
		格式：...azCenter_XXX_XX... 其中 XXX_XX 代表度数（例如，010_22 = 10.22 度）
        """

		# 匹配 azCenter_XXX_XX 或 azCenter_XXXX_XX 等的模式。
        pattern = r'azCenter_(\d+)_(\d+)'
        match = re.search(pattern, filename)
        if match:
            degrees_part = match.group(1)  # e.g., '010'
            decimal_part = match.group(2)  # e.g., '22'
            # 转换为浮点数：度数部分 + 小数部分 / 100
            azimuth = float(degrees_part) + float(decimal_part) / 100.0
            return azimuth
        else:
            # 如果未找到方位角，则返回 0-360 之间的随机方位角。
            print(f"Warning: No azimuth found in filename {filename}, using random azimuth")
            return float(np.random.uniform(0, 360))  # np.random.uniform(0, 360)返回一个numpy.float64标量，外层float()把它转成Python原生的float类型。
```
**文件名解析示例**：
```
2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
                                ^^^^^ ^^
                            degrees=60  decimal=22
                            → azimuth = 60.22°
```

- 用正则从文件名提取方位角（度+百分位）
- 如果提取失败，返回随机角度作为降级方案


## `__len__`数据集大小
```python
    def __len__(self): 
        return len(self.sample_path_label)
```
返回数据集样本总数。`DataLoader` 通过它知道要迭代多少次。


## `__getitem__`获取单个样本
```python
	def __getitem__(self, idx):
	        """
	        参数:
	            idx: 数据索引
	        Returns:
	            sample: 包含 'image'、'name'、'label' 和 'az' 的字典
	        """
	
	        file_path = self.sample_path_label[idx][0]
	        label = self.sample_path_label[idx][1]
	        
	        
	        '''提取不带扩展名的文件名'''
	        # 从完整路径中提取纯文件名
	        filename = os.path.basename(file_path)
	        # 去掉 .png 后缀，取其主名
	        name = os.path.splitext(filename)[0]
	        
	        
	        '''从文件名中提取方位角'''
	        az = self.extract_azimuth_from_filename(filename)
	        
	        
	        '''加载、处理图像'''
	        try:
	            image = Image.open(file_path).convert('L')  #.convert('L')：把图像转为灰度图
	            
	            # np.array(image, dtype=np.float32)：把 PIL Image 转成 numpy 数组，dtype 指定为 float32。灰度图的 shape 是(128, 128)。/ 255.0是归一化至[0.0, 1.0]。
	            img_array = np.array(image, dtype=np.float32) / 255.0
	            
	            # 用户传入自定义处理函数，比如torchvision.transforms.Compose可以做随机裁剪、翻转等数据增强。此时 transform 的输入是numpy 数组，shape(128, 128)。
	            if self.transform:
	                img_array = self.transform(img_array)
	            else:
	                # torch.from_numpy(img_array)：numpy → PyTorch tensor，shape 仍是(128, 128)。.unsqueeze(0)：在第0维插入一个维度，shape 变成(1, 128, 128)，即(C, H, W)格式，1 表示灰度图的单通道
	                img_array = torch.from_numpy(img_array).unsqueeze(0)
	                
	        except Exception as e:
	            print(f"Error loading image {file_path}: {e}")
	            # 如果图片损坏或读取失败，返回一张全黑图和方位角0.0，不让整个训练崩溃
	            img_array = torch.zeros((1, 128, 128))  # 根据需要调整尺寸
	            az = 0.0
	        
	        sample = {
	            'image': img_array,
	            'name': name,
	            'label': label,
	            'az': az
	        }
	        
	        return sample
```

**步骤**：
1. 从 `sample_path_label` 获取文件路径和标签
2. 用 `PIL.Image.open(file_path).convert('L')` 加载为灰度图
3. 转换为 numpy 数组并归一化到 [0, 1]
4. 应用 transform（如 `ToTensor()`）
```
磁盘图片
  │ PIL 读取，转 numpy          ← numpy 用来做像素级别的预处理
  │ np.array() / 255.0           (CPU 上算足够，不需要 GPU)
  │
  │ torch.from_numpy()          ← 转成 tensor，从此进入 PyTorch 世界
  │ .unsqueeze(0)               (GPU 上跑模型必须用 tensor)
  ▼
DDPM 模型（tensor 计算图 → GPU 训练 → 自动求梯度）
```
5. 从文件名提取方位角
6. 返回字典 `{'image', 'name', 'label', 'az'}`
```python
sample = {
    'image': img_array,    # Tensor, (1, 128, 128), 值域 [0,1]
    'name': name,          # 文件名（不含扩展名）
    'label': label,        # int, 0~9
    'az': az               # float, 方位角度数
}
```

### 有关idx

`idx` 是一个**整数索引**，是 Python 的 `__getitem__` 魔术方法的参数。

作用：当你用 `dataset[0]`、`dataset[5]` 这样的方括号语法访问数据集时，`idx` 就是方括号里那个数字。

在这个类中的具体用途
```python
file_path = self.sample_path_label[idx][0]   # 第107行
label = self.sample_path_label[idx][1]        # 第108行
```
`self.sample_path_label` 是一个二维列表，每个元素是 `[文件路径, 标签]` 这样的二元组。`idx` 就是从中选取第几个样本。`[0]、[1]`就代表每个样本的元素。

|表达式|结果|
|---|---|
|`self.sample_path_label[idx]`|取出第 idx 个样本的整个列表，如 `["/path/to/img.png", 3]`|
|`self.sample_path_label[idx][0]`|取该列表的**第0个元素** → 文件路径|
|`self.sample_path_label[idx][1]`|取该列表的**第1个元素** → 标签|

调用者：这个 `__getitem__` 不是手动调用的，而是由 **PyTorch 的 `DataLoader`** 在遍历数据集时自动调用。`DataLoader` 会根据 batch size 和 sampler 自动生成一批索引，逐个传入 `__getitem__` 来加载对应的数据样本。



## 生成数据集 txt 文件
```python
# 创建数据集txt文件的实用函数
def create_sample_txt_file(data_root, output_txt):
    """
	为 SAMPLE 数据集创建一个与 MSTAR 格式相同的 txt 文件
	参数：
		data_root：SAMPLE 数据集的根目录
		output_txt：输出 txt 文件路径
    """

    dataset = SAMPLE_Dataset(data_root=data_root)
    with open(output_txt, 'w') as f:
        for item in dataset.sample_path_label:
            file_path = item[0]
            label = item[1]
            f.write(f"{file_path} {label}\n")
    print(f"Created txt file: {output_txt} with {len(dataset)} samples")

```
**作用**：扫描 `data_root` 目录，将所有图片路径和标签写入一个 txt 文件，供后续用 `txt_file` 方式快速加载。

**使用场景**：首次获得数据集时，目录结构是现成的但还没有 txt 文件。运行这个函数一次性生成 txt，之后训练和推理都用 txt 加载，避免每次都扫描目录。

生成的 txt 格式与 `read_dataset_txt` 读取的格式一致：
```
real/2s1/2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png 0
real/bmp2/bmp2_real_A_elevDeg_016_azCenter_035_49_serial_9563.png 1
...
```


## 模块入口 — 独立运行示例
```python
if __name__ == '__main__':
    # 用法示例
    data_root = r"D:\taojiawei\tjw\Deep_Learning\EM_deeplearning\dataset\SAMPLE\png_images\qpm\real"
    
    # 创建训练集和测试集的txt文件
    create_sample_txt_file(data_root, '../data/sample_train.txt')
    
    # 测试数据集
    dataset = SAMPLE_Dataset(data_root=data_root)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Azimuth: {sample['az']}")
        print(f"Name: {sample['name']}")
```
当直接运行 `python sample_dataset.py` 时执行，功能是：
1. 调用 `create_sample_txt_file` 生成训练集 txt 文件
2. 加载数据集并打印第一个样本的各项属性作为**完整性检查**

**示例输出**：
```
Created txt file: ../data/sample_train.txt with 2747 samples
Dataset size: 2747
Sample keys: dict_keys(['image', 'name', 'label', 'az'])
Image shape: torch.Size([1, 128, 128])
Label: 0
Azimuth: 60.22
Name: 2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01
```