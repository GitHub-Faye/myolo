# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # 声明使用AGPL-3.0开源许可证

# 导入路径操作模块，用于处理文件路径
from pathlib import Path

# 导入基础模型类，YOLO和YOLOWorld类都继承自这个基类
from ultralytics.engine.model import Model
# 导入yolo模块，包含各种任务的具体实现类
from ultralytics.models import yolo
# 导入各种任务的模型类，用于不同类型的计算机视觉任务
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, FacePoseModel, OBBModel, PoseModel, SegmentationModel, WorldModel
# 导入工具函数：ROOT表示项目根目录，yaml_load用于加载YAML配置文件
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) 对象检测模型。"""  # 类文档字符串，简要说明YOLO类的用途

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """
        初始化YOLO模型，如果模型文件名包含'-world'则切换到YOLOWorld模型。
        
        参数:
            model (str): 模型文件路径，默认为"yolo11n.pt"
            task (str): 任务类型，默认为None
            verbose (bool): 是否显示详细信息，默认为False
        """
        # 将模型路径转换为Path对象，便于后续操作
        path = Path(model)
        # 检查模型文件名是否包含'-world'且后缀为.pt、.yaml或.yml
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是YOLOWorld PyTorch模型
            # 创建YOLOWorld实例
            new_instance = YOLOWorld(path, verbose=verbose)
            # 将当前实例的类型替换为YOLOWorld类型
            self.__class__ = type(new_instance)
            # 将当前实例的所有属性替换为YOLOWorld实例的属性
            self.__dict__ = new_instance.__dict__
        else:
            # 如果不是YOLOWorld模型，继续默认的YOLO初始化流程
            # 调用父类Model的初始化方法
            super().__init__(model=model, task=task, verbose=verbose)

    @property  # 使用property装饰器，使方法可以像属性一样被访问
    def task_map(self):
        """
        映射头部到模型、训练器、验证器和预测器类。
        
        返回:
            dict: 包含各种任务对应的组件映射
        """
        # 返回一个字典，键为任务名称，值为该任务对应的各种组件类
        return {
            "classify": {  # 分类任务
                "model": ClassificationModel,  # 分类模型类
                "trainer": yolo.classify.ClassificationTrainer,  # 分类训练器类
                "validator": yolo.classify.ClassificationValidator,  # 分类验证器类
                "predictor": yolo.classify.ClassificationPredictor,  # 分类预测器类
            },
            "detect": {  # 检测任务
                "model": DetectionModel,  # 检测模型类
                "trainer": yolo.detect.DetectionTrainer,  # 检测训练器类
                "validator": yolo.detect.DetectionValidator,  # 检测验证器类
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器类
            },
            "segment": {  # 分割任务
                "model": SegmentationModel,  # 分割模型类
                "trainer": yolo.segment.SegmentationTrainer,  # 分割训练器类
                "validator": yolo.segment.SegmentationValidator,  # 分割验证器类
                "predictor": yolo.segment.SegmentationPredictor,  # 分割预测器类
            },
            "pose": {  # 姿态检测任务
                "model": PoseModel,  # 姿态检测模型类
                "trainer": yolo.pose.PoseTrainer,  # 姿态检测训练器类
                "validator": yolo.pose.PoseValidator,  # 姿态检测验证器类
                "predictor": yolo.pose.PosePredictor,  # 姿态检测预测器类
            },
            "obb": {  # 面向对象边界框任务
                "model": OBBModel,  # OBB模型类
                "trainer": yolo.obb.OBBTrainer,  # OBB训练器类
                "validator": yolo.obb.OBBValidator,  # OBB验证器类
                "predictor": yolo.obb.OBBPredictor,  # OBB预测器类
            },
            "face": {  # 人脸关键点检测任务
                "model": FacePoseModel,  # 人脸关键点检测模型类
                "trainer": yolo.face.FacePoseTrainer,  # 人脸关键点检测训练器类
                "validator": yolo.face.FacePoseValidator,  # 人脸关键点检测验证器类
                "predictor": yolo.face.FacePosePredictor,  # 人脸关键点检测预测器类
            },
        }


class YOLOWorld(Model):
    """YOLO-World 物体检测模型，专注于视觉-语言理解。"""  # 类文档字符串，简要说明YOLOWorld类的用途

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        使用预训练模型文件初始化YOLOv8-World模型。

        加载YOLOv8-World模型用于物体检测。如果没有提供自定义类名，则分配默认的COCO类名。

        参数:
            model (str | Path): 预训练模型文件的路径，支持*.pt和*.yaml格式。
            verbose (bool): 如果为True，在初始化期间打印额外信息。
        """
        # 调用父类Model的初始化方法，指定任务为"detect"
        super().__init__(model=model, task="detect", verbose=verbose)

        # 当模型没有names属性（即没有自定义类名）时，分配默认的COCO类名
        if not hasattr(self.model, "names"):
            # 从COCO8配置文件中加载类名
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property  # 使用property装饰器，使方法可以像属性一样被访问
    def task_map(self):
        """
        映射头部到模型、验证器和预测器类。
        
        返回:
            dict: 包含检测任务对应的组件映射
        """
        # 返回一个字典，只包含detect任务的组件映射
        return {
            "detect": {  # 检测任务
                "model": WorldModel,  # 世界模型类，专门用于视觉-语言理解
                "validator": yolo.detect.DetectionValidator,  # 复用检测验证器类
                "predictor": yolo.detect.DetectionPredictor,  # 复用检测预测器类
                "trainer": yolo.world.WorldTrainer,  # 世界训练器类，专门用于视觉-语言理解模型的训练
            }
        }

    def set_classes(self, classes):
        """
        设置类别。

        参数:
            classes (List(str)): 类别列表，例如 ["person"]。
        """
        # 调用模型的set_classes方法设置类别
        self.model.set_classes(classes)
        
        # 移除背景类（如果存在）
        background = " "  # 定义背景类的名称为空格
        if background in classes:
            # 从类别列表中移除背景类
            classes.remove(background)
        
        # 更新模型的类名属性
        self.model.names = classes

        # 重置预测器的类名
        # self.predictor = None  # 重置预测器，否则旧名称会保留 (此行被注释掉了)
        # 如果预测器存在，则更新其模型的类名
        if self.predictor:
            self.predictor.model.names = classes
