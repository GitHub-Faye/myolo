# Ultralytics �� AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import FacePoseModel
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.models.yolo.face.val import FacePoseValidator


class FacePoseTrainer(BaseTrainer):
    """YOLOv12人脸姿态估计训练器"""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化FacePoseTrainer对象，并指定任务为face"""
        if overrides is None:
            overrides = {}
        overrides["task"] = "face"
        super().__init__(cfg, overrides, _callbacks)
    
    def build_dataset(self, img_path, mode="train", batch=None):
        """构建YOLO数据集"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """构建并返回数据加载器"""
        assert mode in {"train", "val"}, f"Mode必须是'train'或'val'，不能是{mode}。"
        with torch_distributed_zero_first(rank):  # 仅在DDP模式下初始化数据集缓存一次
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True'与DataLoader的shuffle不兼容，设置shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # 返回数据加载器

    def preprocess_batch(self, batch):
        """预处理批次数据，将图像缩放并转换为浮点数"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # 尺寸
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状(拉伸到gs的倍数)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """设置模型属性，包括类别数量、名称和关键点形状"""
        self.model.nc = self.data["nc"]  # 将类别数量附加到模型
        self.model.names = self.data["names"]  # 将类别名称附加到模型
        self.model.args = self.args  # 将超参数附加到模型
        self.model.kpt_shape = self.data["kpt_shape"]  # 将关键点形状附加到模型

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取人脸关键点检测模型"""
        model = FacePoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回适用于人脸关键点检测的验证器"""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return FacePoseValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args),
            _callbacks=self.callbacks
        )
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回带标签的训练损失项字典"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为5位小数的浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回训练进度的格式化字符串，包含轮次、GPU内存、损失、实例和大小"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """绘制包含标注的类标签、边界框和关键点的训练样本"""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """绘制训练/验证指标"""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # 保存results.png 