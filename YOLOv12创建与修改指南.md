# YOLOv12创建与修改指南

## 1. YOLOv12模型简介

YOLOv12是一个注意力机制驱动的实时目标检测模型，由Yunjie Tian、Qixiang Ye和David Doermann开发。该模型引入了创新的区域注意力机制，在保持实时检测速度的同时显著提高了检测精度。

### 1.1 核心特点

- **区域注意力机制**：将特征图分为多个区域进行注意力计算，提高计算效率
- **A2C2f模块**：Area-Attention Cross-stage Channel Fusion，YOLOv12的核心创新模块
- **高效推理**：在T4 GPU上实现毫秒级推理时间
- **多规模模型**：提供从n(nano)到x(xlarge)的多个规模模型，满足不同场景需求

### 1.2 模型规格

| 模型 | 参数量 | FLOPs | 推理时间(T4) | mAP |
|------|--------|-------|-------------|-----|
| YOLOv12n | 2.6M | 6.2G | 1.64ms | 40.6% |
| YOLOv12s | 9.1M | 19.7G | 2.61ms | 48.0% |
| YOLOv12m | 19.7M | 60.4G | 4.86ms | 52.5% |
| YOLOv12l | 26.5M | 83.3G | 6.77ms | 53.7% |
| YOLOv12x | 59.4M | 185.9G | 11.79ms | 55.2% |

## 2. YOLOv12模型配置文件详解

YOLOv12的模型结构和参数由YAML格式的配置文件定义。这些配置文件是构建和修改模型的基础，提供了灵活的模型定制能力。

### 2.1 配置文件基本结构

YOLOv12的核心配置文件位于`ultralytics/cfg/models/v12/yolov12.yaml`，包含以下主要部分：

1. **参数部分**：定义模型的基本参数
2. **缩放配置**：定义不同规模模型的缩放比例
3. **主干网络**：定义特征提取部分
4. **检测头**：定义特征融合和目标检测部分

配置文件示例：

```yaml
# YOLOv12配置文件顶部部分
# YOLOv12 🚀, AGPL-3.0 license
# YOLOv12 object detection model with P3-P5 outputs

# 参数部分
nc: 80  # 类别数量
scales:  # 模型复合缩放常数，例如'model=yolov12n.yaml'将使用'n'缩放
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # 2.6M参数, 6.2 GFLOPs
  s: [0.50, 0.50, 1024]  # 9.1M参数, 19.7 GFLOPs
  m: [0.50, 1.00, 512]   # 19.7M参数, 60.4 GFLOPs
  l: [1.00, 1.00, 512]   # 26.5M参数, 83.3 GFLOPs
  x: [1.00, 1.50, 512]   # 59.4M参数, 185.9 GFLOPs
```

### 2.2 参数详解

#### 2.2.1 基本参数

- **nc**：检测类别数量，默认为80（COCO数据集类别数）。
- **scales**：定义不同规模模型的缩放参数，有三项：
  - 第一项：深度缩放比例（影响模块重复次数）
  - 第二项：宽度缩放比例（影响通道数）
  - 第三项：最大通道数限制

#### 2.2.2 主干网络配置

主干网络定义了模型如何从输入图像提取特征：

```yaml
# YOLO12-turbo backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]]        # 0-P1/2 - 第一个卷积层
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4 - 第二个卷积层
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]       # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]    # 核心区域注意力模块
  - [-1, 1, Conv,  [1024, 3, 2]]      # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]   # 8
```

每一行定义了一个层或模块，格式为：
- **[from, repeats, module, args]**
  - **from**：输入来源。`-1`表示前一层，也可以是具体的层索引。
  - **repeats**：模块重复次数。
  - **module**：模块类型，如`Conv`、`A2C2f`等。
  - **args**：模块参数，根据模块类型不同而变化。

#### 2.2.3 检测头配置

检测头负责特征融合和生成最终检测结果：

```yaml
# YOLO12-turbo head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 上采样
  - [[-1, 6], 1, Concat, [1]]                   # 特征连接
  - [-1, 2, A2C2f, [512, False, -1]]            # 11 - 特征融合

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]            # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]            # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]                 # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]]             # 检测层，使用P3, P4, P5特征
```

特别说明：
- **特征融合**：通过上采样和特征连接（Concat）实现多尺度特征融合。
- **Detect层**：最后一层是检测层，接收多个输入特征层（P3、P4、P5），输出最终的检测框和类别。

### 2.3 核心模块配置详解

#### 2.3.1 A2C2f模块参数

A2C2f是YOLOv12的核心创新模块，配置格式为：
```
[输入层, 重复次数, A2C2f, [通道数, 残差连接, 区域数]]
```

参数解释：
- **通道数**：输出通道数
- **残差连接**：是否使用残差连接（True/False）
- **区域数**：区域注意力中的区域划分数量，影响注意力计算的粒度

例如：`[-1, 4, A2C2f, [512, True, 4]]`表示：
- 输入来自前一层
- 重复4次A2C2f模块
- 输出512通道
- 使用残差连接
- 将特征图划分为4个区域进行注意力计算

#### 2.3.2 C3k2模块参数

C3k2是YOLOv12中使用的改进CSP模块，配置格式为：
```
[输入层, 重复次数, C3k2, [通道数, 残差连接, 通道比例]]
```

例如：`[-1, 2, C3k2, [256, False, 0.25]]`表示：
- 输入来自前一层
- 重复2次C3k2模块
- 输出256通道
- 不使用残差连接
- 内部通道比例为0.25

## 3. 创建YOLOv12模型

### 3.1 从预训练模型加载

使用Ultralytics库可以直接加载预训练的YOLOv12模型：

```python
from ultralytics import YOLO

# 加载预训练的YOLOv12模型
model = YOLO('yolov12n.pt')  # nano版本
# 其他可选规格：yolov12s.pt, yolov12m.pt, yolov12l.pt, yolov12x.pt
```

### 3.2 从配置文件创建模型的详细过程

#### 3.2.1 基本创建步骤

```python
from ultralytics import YOLO

# 从YAML配置文件创建模型
model = YOLO('yolov12.yaml')  # 默认配置
```

这一行代码背后发生了什么：

1. **配置文件解析**：读取YAML文件，解析模型结构和参数
2. **模型实例化**：根据配置创建对应的PyTorch模型
3. **模块构建**：构建backbone和head中定义的各个模块
4. **参数初始化**：初始化模型权重

#### 3.2.2 指定模型规模

可以通过在创建时指定规模参数：

```python
# 创建nano规模的模型
model = YOLO('yolov12n.yaml')

# 或者通过参数指定
model = YOLO('yolov12.yaml', scale='n')
```

内部过程：
1. 读取配置文件中的`scales`部分
2. 根据指定的规模（如'n'）获取对应的缩放参数
3. 对模型深度和宽度进行相应缩放

#### 3.2.3 命令行创建与训练

在命令行中创建和训练模型：

```bash
# 基本命令格式
yolo task=detect mode=train model=yolov12n.yaml data=coco128.yaml epochs=100

# 详细参数示例
yolo task=detect mode=train \
  model=yolov12n.yaml \
  data=coco128.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0
```

命令行参数说明：
- **task**：任务类型，如detect（目标检测）、segment（分割）等
- **mode**：模式，train（训练）、val（验证）、predict（预测）
- **model**：模型配置文件或预训练权重
- **data**：数据集配置文件
- **epochs**：训练轮数
- **batch**：批次大小
- **imgsz**：输入图像尺寸
- **device**：使用的设备（0表示第一个GPU）

### 3.3 配置文件与模型构建的内部机制

当使用`model = YOLO('yolov12.yaml')`创建模型时，内部执行以下步骤：

1. **YAML解析**：
   ```python
   # 伪代码示例
   cfg = yaml.safe_load(open('yolov12.yaml'))
   ```

2. **模型初始化**：
   ```python
   # 伪代码示例
   model = DetectionModel(cfg)  # 创建检测模型实例
   ```

3. **构建网络层**：
   ```python
   # 伪代码示例
   # 根据配置文件中的backbone和head构建网络
   for layer_cfg in cfg['backbone'] + cfg['head']:
       from_layer = layer_cfg[0]
       repeats = layer_cfg[1]
       module_name = layer_cfg[2]
       args = layer_cfg[3]
       
       # 创建并添加模块
       layer = create_module(module_name, args)
       model.add_module(layer)
   ```

4. **应用缩放**：
   ```python
   # 伪代码示例
   if scale in cfg['scales']:
       depth_scale, width_scale, max_channels = cfg['scales'][scale]
       apply_scaling(model, depth_scale, width_scale, max_channels)
   ```

### 3.4 自定义数据集训练

使用自定义数据集训练YOLOv12模型：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov12n.pt')  # 或从配置文件：'yolov12n.yaml'

# 训练模型
results = model.train(
    data='path/to/your/data.yaml',  # 数据集配置文件
    epochs=100,
    batch=16,
    imgsz=640,
    device=0  # 使用的GPU编号
)
```

数据集配置文件例子 (data.yaml):

```yaml
path: /path/to/dataset  # 数据集根目录
train: images/train     # 训练图像目录
val: images/val         # 验证图像目录

# 类别
nc: 3  # 类别数量
names: ['person', 'car', 'bicycle']  # 类别名称
```

## 4. 修改YOLOv12模型

### 4.1 修改模型架构

YOLOv12模型架构定义在配置文件中（`ultralytics/cfg/models/v12/yolov12.yaml`），可以通过修改此文件来自定义模型架构。

#### 4.1.1 创建自定义配置文件

复制原始配置文件并进行修改：

```python
# 创建自定义配置文件
import shutil
from pathlib import Path

# 复制原始配置文件
original_config = Path('ultralytics/cfg/models/v12/yolov12.yaml')
custom_config = Path('custom_yolov12.yaml')
shutil.copy(original_config, custom_config)

# 使用自定义配置文件
model = YOLO('custom_yolov12.yaml')
```

#### 4.1.2 修改主干网络结构

主干网络（backbone）定义了特征提取部分，可以通过修改层数、通道数、模块类型进行自定义：

```yaml
# 修改主干网络示例
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  # 修改A2C2f的区域数量（第三个参数：4→2）
  - [-1, 4, A2C2f, [512, True, 2]]  # 减少区域数量为2
  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8
```

#### 4.1.3 修改检测头结构

检测头（head）处理特征提取后的信息并进行目标检测，可以修改：

```yaml
# 修改检测头示例
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  # 增加A2C2f的重复次数（第二个参数：2→3）
  - [-1, 3, A2C2f, [512, False, -1]] # 增加重复次数

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14
  
  # 后续保持原样...
```

### 4.2 修改区域注意力参数

YOLOv12的核心创新是A2C2f模块中的区域注意力机制。可以通过修改以下参数进行优化：

#### 4.2.1 修改区域数量

区域数量决定了特征图被分割成多少区域进行注意力计算：

```python
# 在配置文件中修改区域数量
- [-1, 4, A2C2f, [512, True, 4]]  # 原始：4个区域
- [-1, 4, A2C2f, [512, True, 2]]  # 修改：2个区域
```

或者在代码中修改A2C2f模块：

```python
# 自定义A2C2f模块
class CustomA2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        # 自定义初始化参数
        self.area = 2  # 修改默认区域数为2
        # 其他初始化代码...
```

#### 4.2.2 修改MLP比率

多层感知机(MLP)比率影响注意力模块的表达能力：

```python
# 在配置文件中添加自定义MLP比率参数
- [-1, 4, A2C2f, [512, True, 4, 1.5]]  # 最后一个参数为自定义MLP比率
```

### 4.3 自定义类别数和任务

#### 4.3.1 修改类别数

对于自定义数据集，需要修改检测的类别数：

```yaml
# 在配置文件中修改类别数
nc: 20  # 原始为80，修改为自己的类别数
```

或在代码中指定：

```python
# 在训练时指定类别数
model.train(data='custom_data.yaml', nc=20)
```

#### 4.3.2 修改任务类型

YOLOv12支持多种计算机视觉任务，可以通过修改任务类型来适应不同需求：

```python
# 检测任务（默认）
model = YOLO('yolov12n.pt')

# 分割任务（需要使用相应的预训练模型）
model = YOLO('yolov12n-seg.pt')

# 关键点检测任务
model = YOLO('yolov12n-pose.pt')
```

### 4.4 高级修改

#### 4.4.1 修改注意力机制

如果需要深度自定义注意力机制，可以修改AAttn和ABlock模块：

```python
# 自定义区域注意力模块
class CustomAAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        # 自定义区域注意力实现
        # ...
        
# 自定义ABlock模块
class CustomABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        # 使用自定义区域注意力
        self.attn = CustomAAttn(dim, num_heads=num_heads, area=area)
        # ...
```

#### 4.4.2 添加新功能模块

可以通过添加新的功能模块来增强YOLOv12的能力：

```python
# 定义新模块
class EnhancedA2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False):
        super().__init__()
        # 增强的A2C2f实现
        # ...

# 在配置文件中使用新模块
backbone:
  # ...
  - [-1, 4, EnhancedA2C2f, [512, True, 4]]  # 使用增强模块
  # ...
```

## 5. 实践示例

### 5.1 创建轻量化YOLOv12模型

创建一个比YOLOv12n更轻量的模型：

```yaml
# 超轻量YOLOv12-nano配置
nc: 80  # 类别数
scales:
  # [depth, width, max_channels]
  nano: [0.25, 0.125, 1024]  # 极小版本

# 其他配置保持不变...
```

### 5.2 高精度YOLOv12模型

创建一个优化精度的YOLOv12模型：

```yaml
# 高精度YOLOv12配置
backbone:
  # [from, repeats, module, args]
  # ...
  # 增加A2C2f的区域数和重复次数
  - [-1, 6, A2C2f, [512, True, 8]]  # 增加区域数和重复次数
  # ...

head:
  # ...
  # 使用更多重复的A2C2f模块
  - [-1, 4, A2C2f, [512, False, 4]]
  # ...
```

### 5.3 训练自定义数据集的完整示例

```python
from ultralytics import YOLO

# 1. 创建或加载模型
model = YOLO('yolov12n.yaml')  # 从配置创建
# model = YOLO('yolov12n.pt')  # 或加载预训练模型

# 2. 训练模型
results = model.train(
    data='custom_data.yaml',  # 数据集配置
    epochs=300,
    batch=64,
    imgsz=640,
    patience=50,  # 早停参数
    optimizer='AdamW',  # 优化器选择
    lr0=0.001,  # 初始学习率
    lrf=0.01,  # 最终学习率因子
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    device=0,
    project='YOLOv12-custom',
    name='run1'
)

# 3. 评估模型
metrics = model.val()

# 4. 导出模型
model.export(format='onnx')  # 导出为ONNX格式
```

## 6. 总结

YOLOv12作为注意力驱动的目标检测模型，提供了强大的性能和灵活的架构。通过本指南中的方法，可以根据具体需求创建和修改YOLOv12模型，实现定制化的目标检测解决方案。无论是提高精度、减小模型体积还是适应特定任务，YOLOv12都提供了丰富的修改空间和可能性。 