# 在Google Colab中使用自定义YOLOv12库

## 预备步骤

1. 登录您的Google账号
2. 打开Google Colab: https://colab.research.google.com/
3. 上传`colab_yolov12.ipynb`文件到Colab

## 使用步骤

1. **确保使用GPU运行环境**
   - 点击菜单 → 修改 → 笔记本设置
   - 硬件加速器选择"GPU"

2. **更新GitHub仓库地址**
   - 将第二个代码块中的`https://github.com/你的用户名/yolov12-main.git`替换为您实际的GitHub仓库地址

3. **运行代码块**
   - 按顺序运行每个代码块（使用播放按钮或按Shift+Enter）
   - 等待每个步骤完成后再运行下一步

4. **故障排查**
   - 如果遇到`ModuleNotFoundError: No module named 'ultralytics'`错误，确保`pip install -e .`命令已成功运行
   - 如果模型权重下载失败，检查网络连接并尝试手动上传权重文件到Colab

## 自定义调整

1. **使用自己的模型权重**
   - 将`model = YOLO('yolov12n.pt')`中的权重路径替换为您的自定义模型路径
   - 您可以上传自己的模型权重到Colab（文件 → 上传到会话存储）

2. **使用自己的测试图像**
   - 上传您自己的图像到Colab
   - 修改代码中的图像路径

3. **自定义训练**
   - 取消注释训练代码块
   - 按需修改训练参数和数据集路径

## 保存工作结果

- 训练结果会保存在`runs/`目录下
- 您可以右键点击文件下载到本地
- 也可以保存到Google Drive（需要挂载）：

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 环境设置

1. 克隆YOLOv12仓库到Colab：
```
!git clone https://github.com/你的用户名/yolov12-main.git
%cd yolov12-main
```

2. 安装依赖项：
```
!pip install -r requirements.txt
```

注意：如果安装flash-attn出现问题，可以尝试以下替代方法：
```
# 对于CUDA 11.8环境
!pip install flash-attn --no-build-isolation

# 或者跳过flash-attn安装
# 修改requirements.txt文件，注释掉flash-attn行
```

3. 使用本地ultralytics：
```
# 从本地文件夹导入YOLO，不需要安装ultralytics包
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov12n.pt')
```

## 常见问题

1. **找不到flash_attn wheel文件**
   - 已将依赖更改为从PyPI安装flash-attn
   - 如果仍有问题，可以尝试直接从源码安装或跳过此依赖

2. **GPU内存不足**
   - 使用较小的模型（如yolov12n.pt）
   - 减小batch_size和图像尺寸

3. **模型文件下载问题**
   - 可以手动下载模型文件，然后上传到Colab

## 示例用法

```python
# 导入并测试YOLOv12
import torch
from ultralytics import YOLO

# 检查GPU
print(f"是否有GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 加载模型
model = YOLO('yolov12n.pt')

# 用于测试的图像
test_image = 'ultralytics/assets/bus.jpg'

# 运行推理
results = model(test_image)

# 显示结果
from IPython.display import display, Image
import cv2

results[0].plot()
cv2.imwrite('results.jpg', results[0].plot())
display(Image('results.jpg'))
```

## 更多信息

有关更多详细信息，请参考：
- [本地ultralytics使用说明.md](本地ultralytics使用说明.md)
- [YOLOv12训练流程指南.md](YOLOv12训练流程指南.md) 