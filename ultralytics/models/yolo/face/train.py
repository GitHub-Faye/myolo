# Ultralytics 🚀 AGPL-3.0 license

from copy import copy
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.face.val import FacePoseValidator

class FacePoseTrainer(PoseTrainer):
    """YOLOv12人脸姿态估计训练器"""
    
    def get_validator(self):
        """返回适用于人脸关键点检测的验证器"""
        self.validator = FacePoseValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args)
        )
        return self.validator
    
    def preprocess_batch(self, batch):
        """预处理批次数据，确保关键点格式正确"""
        batch = super().preprocess_batch(batch)
        # 如果需要针对人脸关键点的特殊处理可在此添加
        return batch 