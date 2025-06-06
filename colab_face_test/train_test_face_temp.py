import json

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "header"
      },
      "source": [
        "# YOLOv12人脸关键点检测模型测试\n",
        "\n",
        "这个笔记本用于测试YOLOv12人脸关键点检测模型的训练和评估。"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "setup"
      },
      "source": [
        "## 1. 环境设置\n",
        "\n",
        "首先克隆YOLOv12仓库并安装依赖"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "clone_repo"
      },
      "source": [
        "# 克隆仓库\n",
        "!git clone https://github.com/YourRepo/yolov12.git\n",
        "!cd yolov12 && pip install -e ."
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dataset_section"
      },
      "source": [
        "## 2. 数据集准备\n",
        "\n",
        "假设我们已经有了处理好的人脸关键点数据集，这里我们将创建一个简单的示例数据"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "dataset_setup"
      },
      "source": [
        "import os\n",
        "import yaml\n",
        "import shutil\n",
        "from pathlib import Path\n",
        "\n",
        "# 创建数据目录结构\n",
        "data_dir = Path('face_data')\n",
        "os.makedirs(data_dir / 'train' / 'images', exist_ok=True)\n",
        "os.makedirs(data_dir / 'train' / 'labels', exist_ok=True)\n",
        "os.makedirs(data_dir / 'val' / 'images', exist_ok=True)\n",
        "os.makedirs(data_dir / 'val' / 'labels', exist_ok=True)\n",
        "\n",
        "# 创建数据集配置文件\n",
        "data_yaml = {\n",
        "    'path': str(data_dir.absolute()),\n",
        "    'train': 'train',\n",
        "    'val': 'val',\n",
        "    'nc': 1,\n",
        "    'names': ['face'],\n",
        "    'kpt_shape': [5, 3],  # 5个关键点，每个点有3个值 [x,y,visible]\n",
        "    'flip_idx': [1, 0, 2, 4, 3],  # 水平翻转时的关键点对应关系\n",
        "    'kpt_names': ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']\n",
        "}\n",
        "\n",
        "with open(data_dir / 'data.yaml', 'w') as f:\n",
        "    yaml.dump(data_yaml, f, default_flow_style=False)\n",
        "\n",
        "print(f\"数据集配置已创建: {data_dir / 'data.yaml'}\")"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "model_training"
      },
      "source": [
        "## 3. 模型训练\n",
        "\n",
        "使用YOLOv12的人脸关键点检测模型进行训练"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "model_training_code"
      },
      "source": [
        "from ultralytics import YOLO\n",
        "\n",
        "# 创建模型 - 使用yolov12-face配置\n",
        "model = YOLO('yolov12n-face.yaml')\n",
        "\n",
        "# 开始训练\n",
        "results = model.train(\n",
        "    data=str(data_dir / 'data.yaml'),  # 数据集配置路径\n",
        "    epochs=100,                        # 训练轮数\n",
        "    imgsz=640,                         # 图像大小\n",
        "    batch=16,                          # 批次大小\n",
        "    patience=20,                       # 早停耐心值\n",
        "    save=True,                         # 保存模型\n",
        "    project='face_runs',               # 项目名称\n",
        "    name='train_face',                 # 运行名称\n",
        "    device=0                           # 使用GPU\n",
        ")"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "model_validation"
      },
      "source": [
        "## 4. 模型验证\n",
        "\n",
        "验证训练完成的模型性能"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "model_validation_code"
      },
      "source": [
        "# 加载最佳模型\n",
        "best_model = YOLO('face_runs/train_face/weights/best.pt')\n",
        "\n",
        "# 在验证集上验证模型\n",
        "val_results = best_model.val(\n",
        "    data=str(data_dir / 'data.yaml'),  # 数据集配置路径\n",
        "    imgsz=640,                         # 图像大小\n",
        "    batch=16,                          # 批次大小\n",
        "    plots=True                         # 生成评估图表\n",
        ")\n",
        "\n",
        "# 打印验证结果\n",
        "print(f\"边界框mAP50: {val_results.box.map50:.4f}\")\n",
        "print(f\"边界框mAP50-95: {val_results.box.map:.4f}\")\n",
        "if hasattr(val_results, 'pose'):\n",
        "    print(f\"关键点mAP50: {val_results.pose.map50:.4f}\")\n",
        "    print(f\"关键点mAP50-95: {val_results.pose.map:.4f}\")"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "model_prediction"
      },
      "source": [
        "## 5. 模型预测\n",
        "\n",
        "使用训练好的模型进行人脸关键点检测预测"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "model_prediction_code"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "from PIL import Image\n",
        "import numpy as np\n",
        "\n",
        "# 加载图像进行预测（使用示例图像）\n",
        "# 实际应用中，替换为你自己的图像路径\n",
        "image_path = 'path/to/test/image.jpg'\n",
        "\n",
        "# 运行预测\n",
        "results = best_model.predict(\n",
        "    source=image_path,\n",
        "    conf=0.25,                 # 置信度阈值\n",
        "    save=True,                 # 保存结果\n",
        "    project='face_runs',       # 项目名称\n",
        "    name='predict_face',       # 运行名称\n",
        "    show_labels=True,          # 显示标签\n",
        "    show_conf=True,            # 显示置信度\n",
        "    line_width=2               # 边界框线宽\n",
        ")\n",
        "\n",
        "# 显示结果\n",
        "for r in results:\n",
        "    im_array = r.plot()\n",
        "    im = Image.fromarray(im_array[..., ::-1])\n",
        "    plt.figure(figsize=(12, 8))\n",
        "    plt.imshow(im)\n",
        "    plt.axis('off')\n",
        "    plt.show()"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "model_export"
      },
      "source": [
        "## 6. 模型导出\n",
        "\n",
        "将训练好的模型导出为不同格式，用于部署"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "model_export_code"
      },
      "source": [
        "# 将模型导出为ONNX格式\n",
        "best_model.export(format='onnx')\n",
        "\n",
        "# 可选：导出为其他格式\n",
        "# 可用格式包括：'torchscript'，'onnx'，'openvino'，'engine'，'coreml'，'saved_model'，'pb'，'tflite'，'edgetpu'，'tfjs'，'paddle'\n",
        "# best_model.export(format='engine', half=True)  # 导出为TensorRT引擎，使用半精度"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "conclusion"
      },
      "source": [
        "## 总结\n",
        "\n",
        "我们已经完成了YOLOv12人脸关键点检测模型的完整工作流程，包括：\n",
        "\n",
        "1. 环境设置\n",
        "2. 数据集准备\n",
        "3. 模型训练\n",
        "4. 模型验证\n",
        "5. 模型预测\n",
        "6. 模型导出\n",
        "\n",
        "这个流程可以用于实际项目中的人脸关键点检测任务。"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.10"
    },
    "colab": {
      "provenance": []
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

# 将笔记本内容写入文件
with open("colab_face_test/train_test_face.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("笔记本文件已生成: colab_face_test/train_test_face.ipynb") 