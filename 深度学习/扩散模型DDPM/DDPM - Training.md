---
title: 训练过程
tags:
  - DDPM
  - Training
related:
  - "[[DDPM - General Framework#7. 模块五：训练管线（Training Pipeline）]]"
---

# 代码1
（代码 1 来源：EM_deeplearning_beifen\DDPM\train_ddpm.py)
```python
import sample_dataset
from ddpm_model import (
    ConditionalUNet,
    make_beta_schedule,
    extract,
    sample_ddpm,
)
```


## parameter_setting(args)：参数解析
```python
def parameter_setting(args):
	# 初始化一个空字典，将作为统一配置容器返回。
    config = {}
    config["bs"] = args.bs              # batch size, 默认 64
    config["lr"] = args.lr              # learning rate, 默认 1e-4
    config["num_epochs"] = args.num_epochs  # 训练轮数, 默认 5000
    config["save_dir"] = args.save_dir   # 保存目录, 默认 "SAMPLE/"
    config["timesteps"] = args.timesteps  # 扩散步数, 默认 1000
    config["beta_schedule"] = args.beta_schedule  # 调度策略, 默认 "linear"

    return config  # 返回字典给调用方（即train_ddpm(config)）。

```
起**参数格式转换**的桥梁作用。
- 接收的 `args` 是 `argparse.parse_args()` 返回的 **Namespace 对象**。
- 可以用 `args.bs`、`args.lr` 这样的点语法访问属性值。

### 为什么要多此一举？
直接把 `args` 传给 `train_ddpm` 不行吗？设计上的考量：
1. 解耦 CLI 与核心逻辑：`train_ddpm` 不依赖 `argparse.Namespace` 类型，只依赖普通字典。如果将来换用 JSON 配置文件、环境变量等启动方式，`train_ddpm` 函数无需修改。
2. 字典可扩展：后续如果需要在 `parameter_setting` 里添加计算出的衍生参数（如路径拼接、设备选择等），可以直接往字典里加，不影响 CLI 接口。
3. 可读写：Namespace 对象的属性可以被覆写，但字典的操作更自然（`config["new_key"] = ...`）。

## build_diffusion()：构建扩散系数
```python
def build_diffusion(timesteps: int, device: torch.device):
	# 计算所有需要的系数:
    betas = make_beta_schedule(timesteps).to(device)  # β_t, (T,)
    alphas = 1.0 - betas   # α_t = 1-β_t, (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏α_s, (T,)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )     # ᾱ_{t-1}, (T,)

	# 关键系数:
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # √ᾱ_t
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # √(1-ᾱ_t)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)   # 1/√α_t

    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
    )   # σ_t² = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)

	# 返回字典包含所有系数，供训练和采样使用:
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }
```

**系数用途速查**：

| 系数                              | 用途                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `sqrt_alphas_cumprod`           | 前向加噪中的原图系数：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| `sqrt_one_minus_alphas_cumprod` | 前向加噪中的噪声系数                                                                    |
| `sqrt_recip_alphas`             | 反向均值计算：$1/\sqrt{\alpha_t}$                                                    |
| `posterior_variance`            | 反向采样时的方差                                                                      |


## train_ddpm(config) ：训练主循环函数
```python
def train_ddpm(config):
    batch_size = config["bs"]
    lr = config["lr"]
    train_epoch = config["num_epochs"]
    timesteps = config["timesteps"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 数据加载
```python
    # data_loader
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_train = sample_dataset.SAMPLE_Dataset(
        txt_file="data/sample_train_90.txt",
        transform=data_transforms,
    )
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
```
- **`transforms.Compose`**：torchvision 提供的**转换管道**，将多个 transform 按顺序串联。数据会依次经过列表中的每个 transform。
	- 这里只包含一个 `transforms.ToTensor()`，作用是把 PIL Image 或 numpy 数组（`H × W × C`）转为 PyTorch tensor（`C × H × W`），同时将像素值从 `[0, 255]` 映射到 `[0.0, 1.0]`。
	- 注意：`SAMPLE_Dataset.__getitem__`（sample_dataset.py:126-127）中，图像在传入 transform 之前已经手动除以 255 归一化到 `[0,1]`，所以 `ToTensor()` 在这里仅做**通道维度重排**（`H×W` → `1×H×W`），不会再改变值域。

- 这里实例化了一个自定义 Dataset，传入两个参数：

| 参数          | 值                                 | 作用          |
| ----------- | --------------------------------- | ----------- |
| `txt_file`  | `"data/sample_train_90.txt"`      | 指定数据索引文件的路径 |
| `transform` | `data_transforms`（即 `ToTensor()`） | 图像预处理管道     |
#### 数据流总览
```
data/sample_train_90.txt
┌─────────────────────────────┐
│ data/2s1/img_001.png 0     │  ← 路径 + 标签
│ data/bmp2/img_032.png 1    │
│ ...                        │
└─────────────────────────────┘
            │
            ▼
    SAMPLE_Dataset.__init__()
    解析为: [[路径,标签], ...]
            │
            ▼
    SAMPLE_Dataset.__getitem__(idx)
    ┌─────────────────────────────┐
    │ 1. 读路径 → 取标签          │
    │ 2. 正则提取方位角 (文件名)   │
    │ 3. PIL打开 → 转灰度(L模式)   │
    │ 4. numpy float32 / 255.0    │
    │ 5. ToTensor() → (1,128,128) │
    │ 6. 返回 {image,label,az}    │
    └─────────────────────────────┘
            │
            ▼
    DataLoader(batch_size=64, shuffle=True)
    堆叠为 batch:
    {
      'image': (64,1,128,128),
      'label': (64,),
      'az':    (64,)
    }
```

#### 这里`SAMPLE_Dataset` 内部做了什么？

对照 sample_dataset.py，分 3 个阶段来看：
##### 阶段1：`__init__` — 初始化时读取索引文件

```python
# sample_dataset.py:29-31
if txt_file:
    self.sample_path_label = self.read_dataset_txt(txt_file)
```
`read_dataset_txt`（38-52行）读取 `data/sample_train_90.txt`，文件内容格式为每行一条记录：
```
path/to/image1.png 3
path/to/image2.png 7
...
```
解析为列表 `[[路径, 标签], [路径, 标签], ...]`，例如 `[["data/2s1/image001.png", 0], [...], ...]`。
- 第一列是图像文件的相对/绝对路径。
- 第二列是类别编号（整数 0~9），对应 10 个 SAR 目标类别：

|编号|类别|含义|
|---|---|---|
|0|`2s1`|自行榴弹炮|
|1|`bmp2`|步兵战车|
|2|`btr70`|装甲运兵车|
|3|`m1`|主战坦克|
|4|`m2`|步兵战车|
|5|`m35`|军用卡车|
|6|`m60`|主战坦克|
|7|`m548`|履带运输车|
|8|`t72`|主战坦克|
|9|`zsu23`|自行高炮|

##### 阶段2：`__getitem__` — 每次取数据时动态处理
```python
# sample_dataset.py:99-145
def __getitem__(self, idx):
    file_path = self.sample_path_label[idx][0]  # 图像路径
    label = self.sample_path_label[idx][1]       # 类别标签

    # 从文件名提取方位角
    az = self.extract_azimuth_from_filename(filename)

    # 加载图像 → 灰度图 → numpy → /255 归一化
    image = Image.open(file_path).convert('L')
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 应用 transform（ToTensor，转为 (1,H,W) 的 tensor）
    if self.transform:
        img_array = self.transform(img_array)

    return {'image': img_array, 'name': name, 'label': label, 'az': az}
```
返回的每个 sample 是字典：
```python
{
    'image': Tensor,   # (1, 128, 128), 值域 [0, 1]
    'name':  str,      # 文件名（无扩展名）
    'label': int,      # 类别 0~9
    'az':    float,    # 方位角（度数）
}
```

##### 阶段3：`extract_azimuth_from_filename` — 方位角提取

```python
# sample_dataset.py:75-94
```

从文件名中通过正则 `azCenter_(\d+)_(\d+)` 提取方位角。例如文件名含 `azCenter_010_22`，解析为 $10 + 0.22 = 10.22°$。
这个角度值会传给训练循环，用于构造 10 维谐波编码，作为模型的条件输入之一。


#### 设计要点
1. **txt 索引文件模式**：不直接在代码里硬编码路径，而是用一个 txt 文件列出所有样本。切换训练集/测试集只需换 txt 文件（如 `sample_train_90.txt` → `sample_test_10.txt`），代码无需改动。
2. **每样本动态加载**：图像不是在 `__init__` 时全读入内存，而是在 `__getitem__` 中按需加载。数据集很大时不会撑爆内存。
3. **方位角作为条件**：不同于普通的类别条件生成（只给类别标签），这里额外提供了 SAR 目标特有的**方位角**信息。模型学会在不同角度生成同一类目标的对应视图，这是 SAR 图像生成的关键特性。


### 模型初始化
```python
    """
    构造参数：
	    img_channels=1：输入/输出通道数，SAR 图像是单通道灰度图
	    base_ch=64：基础通道数，控制模型容量
	    time_dim=256：时间步嵌入的维度
	    cond_dim=64：条件嵌入的中间维度
    """
    model = ConditionalUNet(img_channels=1, base_ch=64, time_dim=256, cond_dim=64).to(
        device
    )

	diffusion = build_diffusion(timesteps, device)  # 扩散参数预计算

    optimizer = optim.Adam(model.parameters(), lr=lr) # 优化器
    mse_loss = nn.MSELoss()  # 损失函数
```
一句话总结：**ConditionalUNet** 定义了可学习的去噪函数，**build_diffusion** 提供了扩散过程的数学参数，**Adam** 驱动参数更新，**MSELoss** 衡量去噪质量——四者构成了 DDPM 训练的完整骨架。

#### 架构全景（对照 ddpm_model.py 111-195行）
```
输入 x_t (B,1,H,W)  +  时间步 t (B,)  +  条件 [class_onehot (B,10), az_vec (B,10)]
         │                      │                        │
         │              SinusoidalPosEmb         torch.cat → (B,20)
         │              → Linear → SiLU          → Linear → SiLU → Linear
         │              → Linear                 → cond_emb (B,64)
         │              → t_emb (B,256)                │
         │                      │                      │
         ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Encoder (编码器 / 下采样路径)                                │
│                                                              │
│  conv_in (1→64)                                              │
│       │                                                      │
│  down1 = ResBlock(64→64)   ← t_emb + cond_emb 注入           │
│       │                                                      │
│  DownSample (stride=2 conv)  → 尺寸减半                      │
│       │                                                      │
│  down2 = ResBlock(64→128)  ← t_emb + cond_emb 注入           │
│       │                                                      │
│  DownSample (stride=2 conv)  → 尺寸减半                      │
│       │                                                      │
│  down3 = ResBlock(128→256) ← t_emb + cond_emb 注入           │
├──────────────────────────────────────────────────────────────┤
│  Bottleneck (瓶颈层)                                          │
│  bot1 = ResBlock(256→256)  ← t_emb + cond_emb 注入           │
├──────────────────────────────────────────────────────────────┤
│  Decoder (解码器 / 上采样路径)                                 │
│                                                              │
│  UpSample (nearest ×2 + conv)                                │
│       │                                                      │
│  cat(up2, d2_skip)  → ResBlock(384→128) ← t_emb + cond_emb  │
│       │                                                      │
│  UpSample (nearest ×2 + conv)                                │
│       │                                                      │
│  cat(up1, d1_skip)  → ResBlock(192→64)  ← t_emb + cond_emb  │
│       │                                                      │
│  conv_out (64→1)                                             │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
   输出 ε̂ (B,1,H,W)  预测的噪声
```
#### 优化器
|组件|说明|
|---|---|
|算法|**Adam**（Adaptive Moment Estimation）|
|优化对象|`model.parameters()` — ConditionalUNet 的所有可训练参数|
|学习率|`lr`（默认 1e-4）|
|其他参数|使用 PyTorch 默认值：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$|

**为什么用 Adam？** DDPM 训练本质上是一个回归任务（预测噪声），Adam 的自适应学习率特性能在不同参数尺度上稳定收敛，是扩散模型训练的标配选择。

#### MSE损失函数
$$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} |\epsilon_i - \hat{\epsilon}_\theta(x_t, t, c_i)|^2 $$

|项目|说明|
|---|---|
|类型|均方误差（Mean Squared Error）|
|默认|`reduction='mean'`，对 batch 内所有元素取平均|
|作用|衡量模型预测的噪声 $\hat{\epsilon}$ 与真实噪声 $\epsilon$ 之间的差异|

这就是 DDPM 原论文中的 **简化训练目标** $\mathcal{L}_{\text{simple}}$：
- 原版变分下界包含复杂的加权系数，Ho et al. 发现去掉权重、只保留 MSE 反而生成质量更好。
- 训练直观：给定加噪图像 $x_t$ 和时间步 $t$，让模型"猜"出加了什么噪声。


### 目录准备与日志记录器
```python
    # results save folder
    root = config["save_dir"]  # 1.设定根目录
    model_name = "SAMPLE_DDPM_"  # 2.设定模型名前缀
    if not os.path.isdir(root):  # 3.创建根目录
        os.mkdir(root)
    if not os.path.isdir(os.path.join(root, "generated_results")):  # 4.创建生成结果子目录
        os.mkdir(os.path.join(root, "generated_results"))

    writer = SummaryWriter("./DDPM")  # 5.TensorBoard日志记录器
```
1. 负责**输出基础设施搭建**：确保保存路径存在、初始化 TensorBoard 日志记录器。
- `config["save_dir"]` 来自命令行参数 `--save_dir`，默认值为 `"SAMPLE/"`

2. 
- 作为生成文件和模型文件的命名前缀。最终文件名形如：
    - `SAMPLE_DDPM_10.png`（第 10 epoch 的生成样本）
    - `SAMPLE_DDPM_ddpm_unet_param.pkl`（最终保存的模型参数）

3. 

| 元素                    | 说明                  |
| --------------------- | ------------------- |
| `os.path.isdir(root)` | 检查路径是否**已存在且是一个目录** |
| `os.mkdir(root)`      | 创建单层目录              |
- **为什么先检查再创建？** 如果目录已存在（比如之前训练过），直接调用 `mkdir` 会抛出 `FileExistsError`。先判断再创建是防御性编程。
- 注意这里用的是 `mkdir`（单层），意味着 `root` 的父目录（如默认值的上级目录 `.`）必须已经存在，否则会报错。

4. 
- **`os.path.join(root, "generated_results")`**：跨平台拼接路径。在 Windows 上结果是 `SAMPLE\generated_results`，Linux 上是 `SAMPLE/generated_results`。
	- 这个子目录专门存放**每 10 epoch 生成的 4×4 样本网格图**。

```
最终目录结构：
SAMPLE/                          ← root
├── generated_results/           ← 生成样本
│   ├── SAMPLE_DDPM_10.png
│   ├── SAMPLE_DDPM_20.png
│   ├── ...
│   └── SAMPLE_DDPM_5000.png
└── SAMPLE_DDPM_ddpm_unet_param.pkl   ← 最终模型
```

5. 

| 元素              | 说明                                                                                  |
| --------------- | ----------------------------------------------------------------------------------- |
| `SummaryWriter` | tensorboardX 库提供的日志写入器，API 与 PyTorch 原生的 `torch.utils.tensorboard.SummaryWriter` 一致 |
| `"./DDPM"`      | 日志文件的写入目录（相对路径，相对于运行脚本的目录）                                                          |

tensorboardX vs torch.utils.tensorboard：功能相同，但 `tensorboardX` 是独立库，不依赖 PyTorch，且在某些版本中更稳定。
在整个训练中的使用（第180行）：
```python
writer.add_scalar("DDPM_loss/train", mean_loss, epoch + 1)
```
每个 epoch 写入一个标量（平均 loss），训练结束后用以下命令查看训练曲线：
```bash
tensorboard --logdir=./DDPM
```


6. 与其他代码块的联系
```
参数解析        目录准备             模型构建
───→ config ──→ root + generated_results/ ──→ model, diffusion, optimizer
                       │
                       ▼
                  SummaryWriter
                       │
                       ▼
                  训练循环 → writer.add_scalar(...) 每 epoch 记录
                       │
                       ▼
                  tensorboard --logdir=./DDPM  查看训练曲线

```

### One-hot 模板
```python
    # label one-hot template
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(
        1, 
        torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 
        1
    )
```
10 个 SAR 目标类别（2s1, bmp2, ..., zsu23）需要以 one-hot 向量形式喂给 U-Net 作为条件。每次都重新构造太浪费——直接预建一个 **10×10 查找表**，用 `labels` 做索引即可 $O(1)$ 取出对应向量。即：
```
初始状态:                          scatter_(dim=1) 操作后:
onehot (10×10 全零)                 onehot (10×10 单位矩阵)

[0,0,0,0,0,0,0,0,0,0]  行0        [1,0,0,0,0,0,0,0,0,0]  ← 索引0→第0列置1
[0,0,0,0,0,0,0,0,0,0]  行1        [0,1,0,0,0,0,0,0,0,0]  ← 索引1→第1列置1
[0,0,0,0,0,0,0,0,0,0]  行2        [0,0,1,0,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0]  行3        [0,0,0,1,0,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0]  行4   →    [0,0,0,0,1,0,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0]  行5        [0,0,0,0,0,1,0,0,0,0]
[0,0,0,0,0,0,0,0,0,0]  行6        [0,0,0,0,0,0,1,0,0,0]
[0,0,0,0,0,0,0,0,0,0]  行7        [0,0,0,0,0,0,0,1,0,0]
[0,0,0,0,0,0,0,0,0,0]  行8        [0,0,0,0,0,0,0,0,1,0]
[0,0,0,0,0,0,0,0,0,0]  行9        [0,0,0,0,0,0,0,0,0,1]
```
scatter_详解见：[[DDPM - Training#`scatter_` 的工作方式：]]
**更简单的等效写法**：`torch.eye(10)` 直接生成 10×10 单位矩阵，

### 训练启动与扩散参数提取
```python
	"""训练启动与初始化"""
    print("DDPM training start!")
    start_time = time.time()  # 记录起始时间


	"""从diffusion字典中提取预计算的扩散参数："""
    betas = diffusion["betas"]
    alphas_cumprod = diffusion["alphas_cumprod"]
    sqrt_alphas_cumprod = diffusion["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = diffusion["sqrt_one_minus_alphas_cumprod"]
```


### 循环训练步骤
```python
	# 外层循环：遍历所有训练轮次（train_epoch是总epoch 数）
    for epoch in range(train_epoch): 
        epoch_start_time = time.time() # 记录当前 epoch 的开始时间
        losses = [] # 初始化一个列表，收集当前 epoch 内所有 batch 的 loss 值，用于后续计算平均 loss
        
		# 内层循环：遍历 DataLoader。从train_loader中逐批取出数据。每个data是一个字典，包含图像、标签和方位角信息。
        for idx, data in enumerate(train_loader):
            model.train() # 将模型设为训练模式（启用 Dropout、BatchNorm 更新等）
            optimizer.zero_grad() # 清空上一轮累积的梯度，防止梯度累加

			"""步骤1: 数据准备与预处理"""
            x0 = data["image"].to(device)  # (B,1,H,W), in [0,1]
            labels = data["label"]  # 类别标签，CPU tensor, (B,)
            az = data["az"].to(device)  # 方位角, (B,)

            # 将图像缩放到 [-1, 1] 范围（图像归一化）
            x0 = x0 * 2.0 - 1.0

			"""步骤2：条件编码"""
            # 类别条件one-hot编码
            class_onehot = onehot[labels].to(device)  # (B,10)

            # 方位角傅里叶编码（10维余弦/正弦，最高至5次谐波）
            real_angle = torch.deg2rad(az) # 将角度从度数转为弧度
            B = x0.shape[0]
            real_az_vec = torch.zeros(B, 10, device=device)
            for i in range(B):
                real_az_vec[i, 0] = torch.cos(real_angle[i])
                real_az_vec[i, 1] = torch.sin(real_angle[i])
                real_az_vec[i, 2] = torch.cos(2 * real_angle[i])
                real_az_vec[i, 3] = torch.sin(2 * real_angle[i])
                real_az_vec[i, 4] = torch.cos(3 * real_angle[i])
                real_az_vec[i, 5] = torch.sin(3 * real_angle[i])
                real_az_vec[i, 6] = torch.cos(4 * real_angle[i])
                real_az_vec[i, 7] = torch.sin(4 * real_angle[i])
                real_az_vec[i, 8] = torch.cos(5 * real_angle[i])
                real_az_vec[i, 9] = torch.sin(5 * real_angle[i])

			"""步骤3：前向扩散（加噪）"""
            # 时间步采样
            t = torch.randint(0, timesteps, (B,), device=device).long()

            noise = torch.randn_like(x0)   # 采样标准正态噪声
            sqrt_alphacum_t = extract(sqrt_alphas_cumprod, t, x0.shape)
            sqrt_one_minus_alphacum_t = extract(
                sqrt_one_minus_alphas_cumprod, t, x0.shape
            )
            # 正向扩散公式：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε（重参数化技巧）
            xt = sqrt_alphacum_t * x0 + sqrt_one_minus_alphacum_t * noise

			"""步骤4 — 模型预测与损失计算"""
            noise_pred = model(xt, t, class_onehot, real_az_vec) # 模型前向传播预测噪声
            loss = mse_loss(noise_pred, noise)  # MSE(预测噪声, 真实噪声)

			"""步骤5：反向传播与参数更新"""
            loss.backward() # 反向传播，计算所有参数的梯度
            optimizer.step() # 根据梯度更新模型参数

            losses.append(loss.item()) # 记录当前batch的loss值（.item()将标量tensor转为Python float）
            
         
		"""Epoch 统计与日志"""
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time  # 计算epoch耗时

        mean_loss = sum(losses) / max(len(losses), 1) # 计算平均 loss
        print(  # 打印日志
            "[Epoch %d/%d] - ptime: %.2f, loss: %.6f"
            % (epoch + 1, train_epoch, per_epoch_ptime, mean_loss)
        )
        # TensorBoard 记录：将平均loss写入TensorBoard，便于可视化训练曲线。
        writer.add_scalar("DDPM_loss/train", mean_loss, epoch + 1)


		"""周期性样本生成，每 10 轮生成一次样本"""
        if (epoch + 1) % 10 == 0 or epoch == train_epoch - 1:
            save_path = os.path.join(  # 拼接保存路径
                root,
                "generated_results",
                model_name + str(epoch + 1) + ".png",
            )
            # 调用采样函数，用当前模型生成 16 张图像
            samples = generate_grid_samples(
                model,
                diffusion,
                device,
            )  # (16,1,128,128), in [-1,1]
            samples = (samples + 1.0) / 2.0  # 反归一化，映射到[0,1]方便保存
            grid = make_grid(samples, nrow=4, normalize=False) # 将16张图排列成4x4的网格图。
            save_image(grid, save_path)  # 保存为 PNG 文件
     
    """训练结束与模型参数保存"""
	print("DDPM Training finish!... save model")
	# 保存模型参数：使用torch.save将模型的状态字典（state_dict）保存为.pkl文件，便于后续加载进行推理或继续训练。
    torch.save(model.state_dict(), os.path.join(root, model_name + "ddpm_unet_param.pkl"))

```


## generate_grid_samples()：生成网格样本函数
```python
@torch.no_grad()
def generate_grid_samples(model, diffusion, device: torch.device):

    """
    diffusion：预计算的扩散参数字典。
	生成 4x4 网格样本（类似于原始 GAN 的 show_result）
	返回形状为 (16, 1, H, W) 的张量，其值在 [-1,1] 范围内
    """

    model.eval()  # 将模型设为评估模式，冻结BatchNorm的running mean/var、关闭 Dropout等训练时特有的行为。采样时必须这样做，否则结果不稳定
    
    # 每个网格（4x4）的样本数量
    n_samples = 16

	"""生成随机条件：类别标签"""
    # 固定标签：从 0 到 9 中随机抽取 n_samples 个标签。随机标签fixed_y：从 0~9 中均匀随机采样 16 个类别标签，每张图一个标签。
    fixed_y = torch.randint(0, 10, (n_samples,))  # (16,)

    onehot = torch.zeros(10, 10)  # 10×10 全零矩阵
    onehot = onehot.scatter_(  # scatter_ 是原地操作
        1,   # dim=1，沿列方向散布
        torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1),  # 每行的目标列索引
        1   # 散步的值
    )
    fixed_y_onehot = onehot[fixed_y]   # 用随机标签作为行索引，从单位矩阵中取出对应的one-hot向量，形状(16, 10)。

	"""生成随机条件：方位角
	目的：为每张生成图像随机指定一个"观察方向"，测试模型能否按不同方位角生成对应的旋转视图
	"""
    fixed_az = torch.deg2rad(360 * torch.rand(n_samples, 1))  # 生成16个在[0°, 360°)内均匀分布的角度，再转为弧度
    fixed_az_vec = torch.zeros(n_samples, 10)
    for i in range(n_samples):  # 谐波编码
        fixed_az_vec[i, 0] = torch.cos(fixed_az[i])
        fixed_az_vec[i, 1] = torch.sin(fixed_az[i])
        fixed_az_vec[i, 2] = torch.cos(2 * fixed_az[i])
        fixed_az_vec[i, 3] = torch.sin(2 * fixed_az[i])
        fixed_az_vec[i, 4] = torch.cos(3 * fixed_az[i])
        fixed_az_vec[i, 5] = torch.sin(3 * fixed_az[i])
        fixed_az_vec[i, 6] = torch.cos(4 * fixed_az[i])
        fixed_az_vec[i, 7] = torch.sin(4 * fixed_az[i])
        fixed_az_vec[i, 8] = torch.cos(5 * fixed_az[i])
        fixed_az_vec[i, 9] = torch.sin(5 * fixed_az[i])

	"""条件张量转移到设备
	将条件向量从 CPU 移到 GPU（如果device='cuda），与模型参数在同一设备上。
	"""
    fixed_y_onehot = fixed_y_onehot.to(device)
    fixed_az_vec = fixed_az_vec.to(device)

	"""提取扩散参数 
	从预计算的字典中提取反向扩散（去噪采样）所需的参数。与训练时只用到正向扩散参数不同，这里多了三项
	"""
    betas = diffusion["betas"]  # 噪声调度
    alphas_cumprod = diffusion["alphas_cumprod"]
    alphas_cumprod_prev = diffusion["alphas_cumprod_prev"]
    sqrt_recip_alphas = diffusion["sqrt_recip_alphas"]
    posterior_variance = diffusion["posterior_variance"]

	"""调用 DDPM 采样
	核心的 DDPM 反向采样函数，执行从纯噪声到图像的完整去噪过程
	"""
    samples = sample_ddpm(
        model,
        betas,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas,
        posterior_variance,  
        fixed_y_onehot,    # 条件信息：类别
        fixed_az_vec,    # 条件信息：方位角编码
        shape=(n_samples, 1, 128, 128),  # 形状：即 16 张单通道 128×128 图像
        device=device,
    )
	
	# 返回值：形状(16, 1, H, W)的tensor，值域[-1, 1]，代表16张生成的图像
    return samples
```

```
# 完整调用链
generate_grid_samples (本函数)
│
├── 随机采样条件 → 16 组 (类别, 方位角)
│   ├── 类别: 0~9 随机 → one-hot (16,10)
│   └── 方位角: [0°,360°) 随机 → 10维谐波编码 (16,10)
│
├── 从 diffusion 字典提取反向采样参数
│
└── 调用 sample_ddpm()
    │
    ├── x_T ← 纯噪声 N(0,I)    形状 (16,1,128,128)
    │
    └── for t = T...1:
        ├── ε̂ = model(x_t, t, class, azimuth)
        ├── x_{t-1} = 去噪一步公式
        └── 循环...
            ↓
        输出 x_0 ∈ [-1,1]

```

作用: 生成 4×4=16 张图像，用于训练过程中的可视化。
步骤：
1. `model.eval()` 切换到评估模式
2. 随机采样 16 个类别标签
3. 随机生成 16 个方位角
4. 构建条件（one-hot + 方位角编码）
5. 调用 `sample_ddpm()` 生成图像
6. 返回 `(16, 1, 128, 128)` 张量，值域 [-1, 1]
核心理念：`generate_grid_samples` 是一个"采样入口"函数，负责准备 16 组随机的类别+视角条件，然后将这些条件和模型一并交给 `sample_ddpm` 完成完整的反向扩散采样。每 10 个 epoch 调用它一次，让你能看到模型生成质量随训练的演变过程。

### `scatter_` 的工作方式：
- 对于第 `i` 行，把 `1` 放到第 `index[i]` 列。
- 最终 `onehot` 就是一个 $10 \times 10$ 的**单位矩阵**：
```
    [[1,0,0,0,0,0,0,0,0,0],   ← 类别 0
     [0,1,0,0,0,0,0,0,0,0],   ← 类别 1
     ...
     [0,0,0,0,0,0,0,0,0,1]]   ← 类别 9
```
- **dim=1**：在第 1 维（列）上操作，即对于每一行，把值放到 `index` 指定的列。
- **index shape 为 `(10,1)`**：10 行，每行一个目标列索引 `[0], [1], ..., [9]`。
- **src=1**：标量 1，被广播到所有指定位置。


### `sample_ddpm(  )` ：
1. 从标准正态分布采样纯噪声 $x_T \sim \mathcal{N}(0, I)$。
2. 从 $t=T$ 到 $t=1$ 逐步去噪，每一步：
    - 用模型预测噪声 $\hat{\epsilon}_\theta(x_t, t, \text{class}, \text{azimuth})$
    - 用公式计算 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_\theta) + \sigma_t z$
3. 最终输出 $x_0$，值域在 $[-1, 1]$。


## main：命令行入口
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="DDPM_training")  # 创建参数解析器,用于解析命令行参数
    
    # 定义命令行参数
    parser.add_argument("--bs", type=int, default=64) # batch size
    parser.add_argument("--lr", type=float, default=1e-4) # 学习率
    parser.add_argument("--num_epochs", type=int, default=5000) # 训练轮数
    parser.add_argument("--save_dir", default="SAMPLE/") # 保存目录
    parser.add_argument("--timesteps", type=int, default=1000) # 扩散步数
    parser.add_argument("--beta_schedule", default="linear") # β调度策略
	
	# 解析命令行参数
	"""
	parse_args()读取 `sys.argv`（即命令行传入的字符串），按上述定义解析为对应的 Python 对象
	返回的 `args` 是一个 `Namespace` 对象，可以用 `args.bs`、`args.lr` 等属性访问各参数值。
	"""
    args = parser.parse_args() 
    
    # 将参数组装为配置字典
    """
    这一步调用了之前定义的 `parameter_setting` 函数，将命令行参数 `args` 转为结构化的配置字典 `config`。
    该函数通常会：
	    1. 根据 `args` 创建目录（如 `save_dir`）。
	    2. 预计算扩散参数（`betas`、`alphas_cumprod` 等）并存入字典。
		3. 初始化 DataLoader、模型、优化器等。
	    4. 返回一个包含所有训练所需对象的字典。
    """
    config = parameter_setting(args)

	# 将整个配置字典传入 `train_ddpm` 函数，正式开始训练流程。
    train_ddpm(config)
```

### 完整启动流程
```
命令行输入:
  python train_ddpm.py --bs 64 --lr 1e-4 --num_epochs 5000 --timesteps 1000

        ↓

if __name__ == "__main__":   ← 判断是否为主程序入口

        ↓

argparse 解析:
  args.bs = 64
  args.lr = 0.0001
  args.num_epochs = 5000
  args.timesteps = 1000
  args.save_dir = "SAMPLE/"
  args.beta_schedule = "linear"

        ↓

parameter_setting(args):
  创建目录 → 预计算扩散参数 → 加载数据 → 构建模型 → 打包为 config 字典

        ↓

train_ddpm(config):
  正式训练 (epoch 循环 → batch 循环 → 加噪 → 去噪预测 → 反向传播)

```