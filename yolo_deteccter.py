#!/usr/bin/env python3
"""
yolo_detector.py - YOLO 检测模块
使用 Sophon SAIL 加载 BModel 进行目标检测

功能：
- 目标检测（YOLOv5/YOLOv8）
- 方向判断
- 距离估算
- 帧率监控
- 日志记录
- 配置管理
"""

import sophon.sail as sail
import cv2
import numpy as np
import time
import json
import os
import logging
from datetime import datetime

# ==================== 配置管理类 ====================
class Config:
    """配置管理类"""
    def __init__(self, config_file=None):
        # 默认配置
        self.conf_thresh = 0.5
        self.nms_thresh = 0.45
        self.frame_skip = 2
        self.alert_cooldown = 2.0
        self.camera_id = 0
        self.bmodel_path = "/data/dataset/bmodels/ten_classes_f32.bmodel"
        self.focal_length = 500      # 相机焦距（像素）
        self.object_height = 0.8     # 物体实际高度（米）
        
        if config_file and os.path.exists(config_file):
            self.load(config_file)
    
    def load(self, config_file):
        """从文件加载配置"""
        with open(config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def save(self, config_file):
        """保存配置到文件"""
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


# ==================== 日志管理类 ====================
class Logger:
    """日志管理类"""
    def __init__(self, log_file='/data/logs/blind_assistant.log', enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('BlindAssistant')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, msg):
        if self.enabled:
            self.logger.info(msg)
    
    def warning(self, msg):
        if self.enabled:
            self.logger.warning(msg)
    
    def error(self, msg):
        if self.enabled:
            self.logger.error(msg)


# ==================== 帧率监控类 ====================
class FPSMonitor:
    """FPS 监控器"""
    def __init__(self):
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
    
    def update(self):
        """更新 FPS"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        return self.fps
    
    def get_fps(self):
        return self.fps
    
    def reset(self):
        """重置计数器"""
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0


# ==================== YOLO 检测器类 ====================
class YOLODetector:
    def __init__(self, bmodel_path=None, conf_thresh=0.5, nms_thresh=0.45, config_file=None):
        """
        初始化 YOLO 检测器
        
        Args:
            bmodel_path: BModel 文件路径
            conf_thresh: 置信度阈值
            nms_thresh: NMS 阈值
            config_file: 配置文件路径
        """
        # 加载配置
        if config_file:
            self.config = Config(config_file)
            conf_thresh = self.config.conf_thresh
            nms_thresh = self.config.nms_thresh
            bmodel_path = bmodel_path or self.config.bmodel_path
        else:
            self.config = Config()
        
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # 初始化日志
        self.logger = Logger()
        
        # 初始化帧率监控
        self.fps_monitor = FPSMonitor()
        
        # 加载 BModel
        self.logger.info(f"加载模型: {bmodel_path}")
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
        
        self.logger.info(f"输入形状: {self.input_shape}")
        self.logger.info(f"输出名称: {self.output_names}")
        print(f"输入形状: {self.input_shape}")
        print(f"输出名称: {self.output_names}")
        
        # 类别名称（根据 data.yaml 设置）
        self.class_names = {
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
        
        # 中文类别名称
        self.class_names_zh = {
            0: '施工标志',
            1: '斑马线',
            2: '禁止通行',
            3: '公交站牌',
            4: '行人',
            5: '汽车',
            6: '自行车',
            7: '黑夜行人',
            8: '黑夜汽车',
            9: '黑夜自行车'
        }
    
    # ==================== 基础函数 ====================
    def sigmoid(self, x):
        """sigmoid 激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def get_fps(self):
        """获取当前 FPS"""
        return self.fps_monitor.get_fps()
    
    def update_fps(self):
        """更新 FPS 统计"""
        return self.fps_monitor.update()
    
    # ==================== 方向判断函数 ====================
    def get_direction(self, bbox, frame_width):
        """
        判断目标在画面中的方向
        
        Args:
            bbox: [x1, y1, x2, y2] 检测框坐标
            frame_width: 画面宽度
        
        Returns:
            direction: '左前方' | '正前方' | '右前方'
        """
        x1, x2 = bbox[0], bbox[2]
        center = (x1 + x2) / 2
        
        if center < frame_width / 3:
            return "左前方"
        elif center > frame_width * 2 / 3:
            return "右前方"
        else:
            return "正前方"
    
    # ==================== 距离估算函数 ====================
    def estimate_distance(self, bbox, depth_map=None, focal_length=None, object_height=None):
        """
        根据检测框大小估算物体距离
        
        Args:
            bbox: [x1, y1, x2, y2] 检测框坐标
            depth_map: 深度图（可选）
            focal_length: 相机焦距（像素）
            object_height: 物体实际高度（米）
        
        Returns:
            distance: 距离（米），-1 表示无法估算
        """
        focal_length = focal_length or self.config.focal_length
        object_height = object_height or self.config.object_height
        
        # 方法1: 基于深度图
        if depth_map is not None:
            try:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # 确保坐标在范围内
                h, w = depth_map.shape[:2]
                center_x = max(0, min(w-1, center_x))
                center_y = max(0, min(h-1, center_y))
                distance = depth_map[center_y, center_x]
                if distance > 0:
                    return round(float(distance), 1)
            except Exception as e:
                self.logger.warning(f"深度图距离估算失败: {e}")
        
        # 方法2: 基于单目视觉
        x1, y1, x2, y2 = bbox
        box_height = y2 - y1
        if box_height > 0:
            distance = (object_height * focal_length) / box_height
            return round(distance, 1)
        
        return -1
    
    # ==================== 解码函数 ====================
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
    
    # ==================== NMS 函数 ====================
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
    
    # ==================== 预处理和后处理 ====================
    def preprocess(self, img):
        """图像预处理"""
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        return img_input
    
    # ==================== 检测函数 ====================
    def detect(self, img, with_info=False):
        """
        检测图像中的目标
        
        Args:
            img: OpenCV 图像 (BGR格式)
            with_info: 是否返回方向、距离等信息
        
        Returns:
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, class_id, confidence]
            detections_info: (可选) 包含方向、距离的详细信息
        """
        # 更新帧率
        fps = self.update_fps()
        
        # 预处理
        img_input = self.preprocess(img)
        
        # 推理
        outputs = self.engine.process(self.graph_name, {self.input_name: img_input})
        
        # 解码
        if 'output0_Concat' in outputs:
            detections = self.decode_yolov8(outputs['output0_Concat'], img.shape)
        else:
            detections = self.decode_yolov5(outputs, img.shape)
        
        # NMS
        detections = self.nms(detections)
        
        # 可选：添加详细信息
        if with_info:
            h_img, w_img = img.shape[:2]
            detections_info = []
            for det in detections:
                x1, y1, x2, y2, cls_id, conf = det
                direction = self.get_direction([x1, y1, x2, y2], w_img)
                distance = self.estimate_distance([x1, y1, x2, y2])
                detections_info.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(cls_id),
                    'class_name': self.class_names.get(int(cls_id), 'unknown'),
                    'class_name_zh': self.class_names_zh.get(int(cls_id), '未知'),
                    'confidence': float(conf),
                    'direction': direction,
                    'distance': distance,
                    'fps': fps
                })
            return detections_info
        
        return detections
    
    def detect_from_file(self, image_path, with_info=False):
        """从文件检测"""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"无法读取图像: {image_path}")
            return []
        return self.detect(img, with_info)
    
    # ==================== 辅助函数 ====================
    def get_class_name(self, class_id, lang='en'):
        """获取类别名称"""
        if lang == 'zh':
            return self.class_names_zh.get(class_id, f'class_{class_id}')
        return self.class_names.get(class_id, f'class_{class_id}')
    
    def set_conf_thresh(self, thresh):
        """设置置信度阈值"""
        self.conf_thresh = thresh
        self.logger.info(f"置信度阈值设置为: {thresh}")
    
    def set_nms_thresh(self, thresh):
        """设置 NMS 阈值"""
        self.nms_thresh = thresh
        self.logger.info(f"NMS 阈值设置为: {thresh}")
    
    # ==================== 可视化函数 ====================
    def draw_detections(self, img, detections, show_info=True):
        """
        在图像上绘制检测结果
        
        Args:
            img: 原始图像
            detections: 检测结果（带信息版本）
            show_info: 是否显示方向和距离信息
        
        Returns:
            img: 绘制后的图像
        """
        colors = {
            0: (0, 0, 255),    # 施工标志 - 红色
            1: (0, 255, 0),    # 斑马线 - 绿色
            2: (255, 0, 0),    # 禁止通行 - 蓝色
            3: (255, 255, 0),  # 公交站牌 - 青色
            4: (0, 255, 255),  # 行人 - 黄色
            5: (255, 0, 255),  # 汽车 - 品红
            6: (128, 128, 0),  # 自行车 - 橄榄绿
            7: (0, 128, 128),  # 黑夜行人 - 墨绿
            8: (128, 0, 128),  # 黑夜汽车 - 紫色
            9: (128, 128, 128) # 黑夜自行车 - 灰色
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            name = det.get('class_name_zh', det.get('class_name', 'unknown'))
            
            color = colors.get(cls_id, (0, 255, 0))
            
            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if show_info and 'direction' in det and 'distance' in det:
                label = f"{name}: {conf:.2f} | {det['direction']} {det['distance']:.1f}m"
            else:
                label = f"{name}: {conf:.2f}"
            
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制 FPS
        fps = self.get_fps()
        cv2.putText(img, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img


# ==================== 多模型检测类 ====================
class MultiModelDetector:
    """多模型检测器（融合多个检测结果）"""
    
    def __init__(self):
        self.detectors = {}
    
    def add_detector(self, name, detector):
        """添加检测器"""
        self.detectors[name] = detector
    
    def detect(self, img, with_info=False):
        """使用所有模型检测并融合结果"""
        all_detections = []
        
        for name, detector in self.detectors.items():
            detections = detector.detect(img, with_info)
            if with_info:
                for det in detections:
                    det['model'] = name
                all_detections.extend(detections)
            else:
                all_detections.extend(detections)
        
        return all_detections


# ==================== 测试代码 ====================
if __name__ == '__main__':
    import sys
    
    # 配置
    BMODEL_PATH = "/data/dataset/bmodels/ten_classes_f32.bmodel"
    TEST_IMAGE = "/data/dataset/runs/detect/盲道.webp"
    
    # 创建检测器
    print("=" * 50)
    print("YOLO Detector 测试")
    print("=" * 50)
    
    detector = YOLODetector(BMODEL_PATH, conf_thresh=0.5, nms_thresh=0.45)
    
    # 测试单张图片
    print(f"\n检测图片: {TEST_IMAGE}")
    results = detector.detect_from_file(TEST_IMAGE, with_info=True)
    
    print(f"\n检测到 {len(results)} 个目标:")
    for det in results:
        print(f"  [{det['class_name_zh']}] {det['confidence']:.2f} | "
              f"{det['direction']} | 距离: {det['distance']:.1f}m | "
              f"位置: {det['bbox']}")
    
    # 测试摄像头
    print("\n尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = detector.detect(frame, with_info=True)
            print(f"实时检测到 {len(results)} 个目标")
        cap.release()
    else:
        print("摄像头不可用")
    
    # 测试 FPS
    print(f"\n当前 FPS: {detector.get_fps()}")
    
    print("\n测试完成！")
