# 使用本地ultralytics库的说明

本项目包含完整的ultralytics库代码，位于项目根目录的`ultralytics/`文件夹中。所有代码已经被修改为使用本地版本的ultralytics，而不是通过pip安装的包。

## 优势

使用本地版本的ultralytics有以下优势：
1. 可以直接修改和自定义ultralytics库的功能
2. 确保项目使用的是兼容的ultralytics版本
3. 使项目可以在没有网络连接的环境中运行
4. 避免依赖外部包版本变更带来的兼容性问题

## 使用方法

在Python代码中导入ultralytics时，使用以下方式：

```python
# 正确的导入方式
from ultralytics import YOLO  # 从本地文件夹导入
```

### 创建或加载模型

对于人脸检测任务，我们已经添加了FacePose模块的支持：

```python
# 加载预训练模型
model = YOLO('yolov12n.pt')  # 检测模型
model = YOLO('yolov12n-seg.pt')  # 分割模型
model = YOLO('yolov12n-pose.pt')  # 姿态检测模型

# 人脸检测模型（已添加FacePose模块支持）
model = YOLO('yolov12n-face.pt', task='face')  # 人脸检测预训练模型
model = YOLO('ultralytics/cfg/models/v12/yolov12-face.yaml', task='face')  # 从配置创建face模型
```

## 依赖安装

虽然不需要安装ultralytics包，但仍需安装其依赖项：

```bash
pip install -r requirements.txt
```

### flash-attn安装问题

如果安装flash-attn出现问题，可以尝试：

```bash
# 对于CUDA 11.8环境
pip install flash-attn --no-build-isolation

# 或者跳过flash-attn安装（在requirements.txt中注释掉相关行）
```

## 项目结构

- `ultralytics/`: 完整的ultralytics库源代码
  - `ultralytics/nn/modules/head.py`: 包含FacePose等检测头的定义
  - `ultralytics/cfg/models/v12/yolov12-face.yaml`: 人脸检测模型配置
- `app.py`: 使用本地ultralytics的应用程序示例
- `colab_setup_yolov12.py`: Colab环境设置脚本
- `face_model_example.py`: 人脸检测模型使用示例

## 注意事项

如果您需要更新ultralytics库，您应该直接更新项目中的`ultralytics/`文件夹，而不是通过pip更新包。

## 自定义模块说明

### FacePose模块

我们添加了FacePose模块用于人脸检测和关键点识别：
- 位置：`ultralytics/nn/modules/head.py`
- 功能：同时检测人脸并预测5个面部关键点
- 使用方法：指定任务类型为'face'，并在yaml配置中使用FacePose头部

## 常见问题解决

1. **模型加载错误**：确保正确指定task参数，特别是对于face、pose等特殊任务
2. **模块找不到**：检查是否正确使用了本地ultralytics库
3. **依赖问题**：确保安装了所有必要的依赖项 