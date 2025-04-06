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