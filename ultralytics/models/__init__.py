# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .fastsam import FastSAM
from .nas import NAS
from .rtdetr import RTDETR
from .sam import SAM
from .yolo import YOLO, YOLOWorld
from .yolo.face import FacePoseModel  # 添加FacePoseModel导入

__all__ = "YOLO", "RTDETR", "SAM", "FastSAM", "NAS", "YOLOWorld"  # allow simpler import

import torch

# 如果有face相关导入的注释，取消注释或添加新导入
from .yolo.classify import ClassificationModel
from .yolo.detect import DetectionModel
from .yolo.pose import PoseModel
from .yolo.segment import SegmentationModel
from .yolo.obb import OBBModel

# 添加face到任务映射
TASK_MAP = {
    "classify": ClassificationModel,
    "detect": DetectionModel,
    "pose": PoseModel,
    "segment": SegmentationModel,
    "obb": OBBModel,
    "face": FacePoseModel,  # 添加face任务
}

def get_model(cfg, task=None, verbose=True):
    """
    创建并返回基于配置和任务的Ultralytics模型。

    Args:
        cfg (str): 配置文件或预训练权重路径，如'yolov8n.yaml'或'yolov8n.pt'
        task (str): 任务类型，如'detect'或'segment'。当None时，从cfg自动推断
        verbose (bool): 是否输出详细信息。默认为True
        
    Returns:
        模型: 从配置创建的适用于指定任务的Ultralytics模型
    """
    model = torch.hub.load("ultralytics/ultralytics", "custom", cfg, auth_token=None, verbose=verbose, trust_repo=True)
    if task is not None:
        model.task = task
    return model
