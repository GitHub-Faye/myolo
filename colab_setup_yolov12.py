"""
设置和使用自定义YOLOv12库的Colab脚本
"""

# 克隆YOLOv12仓库
# !git clone https://github.com/你的用户名/yolov12-main.git
# %cd yolov12-main

# 安装依赖，但不包括ultralytics（使用本地版本）
# !pip install -r requirements.txt

# 不需要安装此项目作为包，因为我们直接使用本地ultralytics文件夹
# 注释掉下面的行：
# !pip install -e .

# 导入并测试YOLOv12
import torch
# 从本地ultralytics文件夹导入YOLO，而不是从安装的包导入
from ultralytics import YOLO

# 检查是否有可用的GPU
print(f"是否有GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 加载模型
model = YOLO('yolov12n.pt')  # 或者您的自定义模型路径

# 用于测试的图像路径
test_image = 'assets/bus.jpg'  # 如果有测试图像，请使用实际路径

# 运行推理
results = model(test_image)

# 显示结果
from IPython.display import display, Image
import cv2
import numpy as np

# 保存和显示结果图像
results[0].plot()
cv2.imwrite('results.jpg', results[0].plot())
display(Image('results.jpg')) 