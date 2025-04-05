# Ultralytics 🚀 AGPL-3.0 license

from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import ops
import numpy as np

class FacePoseValidator(PoseValidator):
    """人脸关键点验证器"""

    def get_desc(self):
        """返回评估指标的描述字符串"""
        return ("%22s" + "%11s" * 10) % (
            "类别",
            "图像数",
            "实例数",
            "边界框(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "关键点(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model):
        """初始化人脸关键点检测的评估指标"""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        # 人脸关键点的sigma值，可以根据实际需要调整
        nkpt = self.kpt_shape[0]
        # 为不同的人脸关键点设置不同的OKS sigma值
        face_sigmas = np.array([0.35, 0.35, 0.25, 0.35, 0.35])  # 眼睛、鼻子、嘴角的权重
        self.sigma = face_sigmas if len(face_sigmas) == nkpt else np.ones(nkpt) / nkpt
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def _prepare_batch(self, si, batch):
        """准备一个批次的数据，将关键点转换为实际坐标"""
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch 