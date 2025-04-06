#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人脸关键点检测示例脚本
使用YOLOv12 face模型进行人脸检测和关键点识别
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12面部关键点检测演示')
    parser.add_argument('--source', type=str, default='0', help='视频源，可以是0(网络摄像头)或视频文件路径')
    parser.add_argument('--model', type=str, default='yolov12n-face.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--show', action='store_true', help='显示检测结果')
    parser.add_argument('--save', action='store_true', help='保存检测结果')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    return parser.parse_args()

def draw_face_keypoints(img, results, conf_threshold=0.25):
    """
    绘制人脸关键点
    
    参数:
        img: 输入图像
        results: YOLO模型预测结果
        conf_threshold: 置信度阈值
    
    返回:
        标注后的图像
    """
    # 关键点连接配置
    connections = [(0, 1), (0, 2), (1, 2), (3, 4)]  # 眼睛和嘴巴连接
    
    # 关键点颜色
    colors = [
        (255, 0, 0),   # 左眼 - 蓝色
        (0, 0, 255),   # 右眼 - 红色
        (0, 255, 0),   # 鼻子 - 绿色
        (255, 0, 255), # 左嘴角 - 紫色
        (0, 255, 255)  # 右嘴角 - 黄色
    ]
    
    # 关键点名称
    kpt_names = ['左眼', '右眼', '鼻子', '左嘴角', '右嘴角']
    
    # 复制原图像
    annotated_img = img.copy()
    
    # 遍历每个检测结果
    for det in results:
        # 获取边界框
        boxes = det.boxes
        
        # 绘制每个检测到的人脸
        for i, box in enumerate(boxes):
            # 获取置信度
            conf = float(box.conf)
            
            # 如果置信度大于阈值
            if conf > conf_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 绘制边界框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加置信度文本
                cv2.putText(
                    annotated_img, 
                    f'人脸 {conf:.2f}', 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
                
                # 获取关键点
                keypoints = det.keypoints
                if keypoints is not None:
                    # 获取当前人脸的关键点
                    kpts = keypoints[i].data[0].cpu().numpy()
                    
                    # 绘制每个关键点
                    for j, kpt in enumerate(kpts):
                        x, y, visible = kpt
                        if visible > 0.5:  # 只绘制可见的关键点
                            # 绘制关键点
                            cv2.circle(annotated_img, (int(x), int(y)), 5, colors[j], -1)
                            
                            # 添加关键点名称
                            cv2.putText(
                                annotated_img, 
                                kpt_names[j], 
                                (int(x), int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                colors[j], 
                                1
                            )
                    
                    # 绘制关键点连接
                    for connection in connections:
                        pt1, pt2 = connection
                        if kpts[pt1][2] > 0.5 and kpts[pt2][2] > 0.5:  # 确保两个关键点都可见
                            cv2.line(
                                annotated_img,
                                (int(kpts[pt1][0]), int(kpts[pt1][1])),
                                (int(kpts[pt2][0]), int(kpts[pt2][1])),
                                (255, 255, 255),
                                2
                            )
    
    return annotated_img

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载模型
    model = YOLO(args.model)
    
    # 打开视频源
    cap = cv2.VideoCapture(0 if args.source == '0' else args.source)
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入器
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 处理每一帧
    while cap.isOpened():
        # 读取一帧
        success, frame = cap.read()
        if not success:
            break
        
        # 运行预测
        results = model.predict(frame, conf=args.conf, verbose=False)
        
        # 绘制关键点
        annotated_frame = draw_face_keypoints(frame, results, args.conf)
        
        # 显示处理后的帧
        if args.show:
            cv2.imshow('YOLOv12 Face Keypoints', annotated_frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 保存处理后的帧
        if args.save and writer is not None:
            writer.write(annotated_frame)
    
    # 释放资源
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print('处理完成!')

if __name__ == '__main__':
    main() 