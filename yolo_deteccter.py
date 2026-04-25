#!/usr/bin/env python3
"""
yolo_detector.py - YOLO 检测模块
使用 Sophon SAIL 加载 BModel 进行目标检测
"""

import sophon.sail as sail
import cv2
import numpy as np
import time

class YOLODetector:
    def __init__(self, bmodel_path, conf_thresh=0.5, nms_thresh=0.45):
        """
        初始化 YOLO 检测器
        
        Args:
            bmodel_path: BModel 文件路径
            conf_thresh: 置信度阈值
            nms_thresh: NMS 阈值
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # 加载 BModel
        print(f"加载模型: {bmodel_path}")
        self.engine = sail.Engine(bmodel_path, 0, sail.IOMode.SYSIO)
        
        # 获取模型信息
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.output_names = self.engine.get_output_names(self.graph_name)
        
        # 解析输入输出信息
        self.batch_size, self.channels, self.input_h, self.input_w = self.input_shape
        self.num_classes = None  # 将在第一次推理时确定
        
        print(f"输入形状: {self.input_shape}")
        print(f"输出名称: {self.output_names}")
        
    def sigmoid(self, x):
        """sigmoid 激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def decode_yolov8(self, output, img_shape):
        """
        解码 YOLOv8 输出
        
        Args:
            output: 模型输出，形状为 (1, num_classes+4, num_boxes)
            img_shape: 原始图像形状 (height, width)
        
        Returns:
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, class_id, confidence]
        """
        h_img, w_img = img_shape[:2]
        
        # 解析输出维度
        _, feat_dim, num_boxes = output.shape
        self.num_classes = feat_dim - 4
        
        detections = []
        
        for i in range(num_boxes):
            # 获取边界框坐标 (归一化的中心点 + 宽高)
            cx = output[0, 0, i]
            cy = output[0, 1, i]
            bw = output[0, 2, i]
            bh = output[0, 3, i]
            
            # 目标置信度
            conf = self.sigmoid(output[0, 4, i])
            if conf < self.conf_thresh:
                continue
            
            # 类别分数
            cls_scores = output[0, 5:self.num_classes+5, i]
            cls_id = np.argmax(cls_scores)
            cls_conf = self.sigmoid(cls_scores[cls_id])
            
            # 综合置信度
            score = conf * cls_conf
            if score < self.conf_thresh:
                continue
            
            # 转换到原图坐标
            x1 = (cx - bw / 2) * w_img
            y1 = (cy - bh / 2) * h_img
            x2 = (cx + bw / 2) * w_img
            y2 = (cy + bh / 2) * h_img
            
            # 裁剪到图像边界内
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_img, int(x2))
            y2 = min(h_img, int(y2))
            
            if x2 > x1 and y2 > y1:
                detections.append([x1, y1, x2, y2, cls_id, score])
        
        return detections
    
    def decode_yolov5(self, outputs, img_shape):
        """
        解码 YOLOv5 输出（3个输出层）
        
        Args:
            outputs: 模型输出字典
            img_shape: 原始图像形状
        
        Returns:
            detections: 检测结果列表
        """
        # YOLOv5 的 anchor 参数（根据你的模型调整）
        anchors = [
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)]
        ]
        strides = [8, 16, 32]
        
        h_img, w_img = img_shape[:2]
        detections = []
        
        for idx, out_name in enumerate(self.output_names):
            out = outputs[out_name]
            _, num_anchors, grid_h, grid_w, _ = out.shape
            
            for a in range(num_anchors):
                for i in range(grid_h):
                    for j in range(grid_w):
                        # 目标置信度
                        conf = self.sigmoid(out[0, a, i, j, 4])
                        if conf < self.conf_thresh:
                            continue
                        
                        # 类别分数
                        cls_scores = out[0, a, i, j, 5:]
                        cls_id = np.argmax(cls_scores)
                        cls_conf = self.sigmoid(cls_scores[cls_id])
                        score = conf * cls_conf
                        
                        if score < self.conf_thresh:
                            continue
                        
                        # 解码边界框
                        tx = out[0, a, i, j, 0]
                        ty = out[0, a, i, j, 1]
                        tw = out[0, a, i, j, 2]
                        th = out[0, a, i, j, 3]
                        
                        cx = (self.sigmoid(tx) * 2 - 0.5 + j) * strides[idx]
                        cy = (self.sigmoid(ty) * 2 - 0.5 + i) * strides[idx]
                        w = (self.sigmoid(tw) * 2) ** 2 * anchors[idx][a][0]
                        h = (self.sigmoid(th) * 2) ** 2 * anchors[idx][a][1]
                        
                        x1 = (cx - w / 2) * w_img / self.input_w
                        y1 = (cy - h / 2) * h_img / self.input_h
                        x2 = (cx + w / 2) * w_img / self.input_w
                        y2 = (cy + h / 2) * h_img / self.input_h
                        
                        detections.append([int(x1), int(y1), int(x2), int(y2), cls_id, score])
        
        return detections
    
    def nms(self, detections):
        """非极大值抑制"""
        if len(detections) == 0:
            return []
        
        detections = np.array(detections)
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 5]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            idx = np.where(iou <= self.nms_thresh)[0]
            order = order[idx + 1]
        
        return detections[keep].tolist()
    
    def preprocess(self, img):
        """图像预处理"""
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        return img_input
    
    def detect(self, img):
        """
        检测图像中的目标
        
        Args:
            img: OpenCV 图像 (BGR格式)
        
        Returns:
            detections: 检测结果列表
        """
        # 预处理
        img_input = self.preprocess(img)
        
        # 推理
        outputs = self.engine.process(self.graph_name, {self.input_name: img_input})
        
        # 解码
        if 'output0_Concat' in outputs:
            # YOLOv8 格式
            detections = self.decode_yolov8(outputs['output0_Concat'], img.shape)
        else:
            # YOLOv5 格式
            detections = self.decode_yolov5(outputs, img.shape)
        
        # NMS
        final_detections = self.nms(detections)
        
        return final_detections
    
    def detect_from_file(self, image_path):
        """从文件检测"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []
        return self.detect(img)
    
    def get_class_name(self, class_id):
        """获取类别名称（需要根据你的模型设置）"""
        # 这里需要根据你的 data.yaml 设置
        names = {
            0: 'construction_sign',
            1: 'zebra_crossing',
            2: 'no_entry',
            3: 'bus_stop',
            4: 'r_person',
            5: 'r_car',
            6: 'r_bicycle',
            7: 'd_person',
            8: 'd_car',
            9: 'd_bicycle'
        }
        return names.get(class_id, f'class_{class_id}')


# 测试代码
if __name__ == '__main__':
    import sys
    
    # 配置
    BMODEL_PATH = "/data/dataset/bmodels/ten_classes_f32.bmodel"
    TEST_IMAGE = "/data/dataset/runs/detect/盲道.webp"
    
    # 创建检测器
    detector = YOLODetector(BMODEL_PATH, conf_thresh=0.5, nms_thresh=0.45)
    
    # 检测图片
    print(f"\n检测图片: {TEST_IMAGE}")
    results = detector.detect_from_file(TEST_IMAGE)
    
    print(f"检测到 {len(results)} 个目标:")
    for det in results:
        x1, y1, x2, y2, cls_id, conf = det
        class_name = detector.get_class_name(int(cls_id))
        print(f"  {class_name}: {conf:.2f} [{x1}, {y1}, {x2}, {y2}]")
    
    # 实时检测测试
    print("\n尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = detector.detect(frame)
            print(f"实时检测到 {len(results)} 个目标")
        cap.release()
    else:
        print("摄像头不可用")
