# YOLOv12 Face 模型示例
# 此示例展示如何使用本地ultralytics库加载face模型

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 检查是否有可用的GPU
print(f"是否有GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 方法1: 加载预先训练好的face模型
# model = YOLO('yolov12n-face.pt', task='face')  # 如果您有预训练的权重文件

# 方法2: 从YAML配置创建新的face模型
# 现在我们已经将FacePose模块添加到了exports中，可以直接使用FacePose模块
model = YOLO('ultralytics/cfg/models/v12/yolov12-face.yaml', task='face')

# 准备一个测试图像 (使用您自己的人脸图像路径)
image_path = 'ultralytics/assets/bus.jpg'  # 替换为您自己的图像
image = cv2.imread(image_path)

if image is None:
    print(f"无法读取图像: {image_path}")
    # 尝试使用另一个示例图像
    image_path = 'ultralytics/assets/zidane.jpg'
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像。请检查图像路径是否正确，并确保图像文件存在。")

# 运行推理
results = model(image, verbose=False)

# 显示结果
result_image = results[0].plot()
cv2.imwrite('face_detection_result.jpg', result_image)
print(f"结果已保存至 face_detection_result.jpg")

# 打印检测到的结果
for r in results:
    print(f"检测到 {len(r.boxes)} 个人脸")
    
    # 如果检测到人脸，打印关键点信息
    if r.keypoints is not None and len(r.keypoints) > 0:
        print(f"检测到 {len(r.keypoints)} 组关键点")
        for i, kpts in enumerate(r.keypoints.data):
            print(f"  人脸 #{i+1} 关键点:")
            for j, kpt in enumerate(kpts):
                x, y, conf = kpt.tolist()
                print(f"    关键点 #{j+1}: x={x:.1f}, y={y:.1f}, 置信度={conf:.2f}")

print("完成!") 