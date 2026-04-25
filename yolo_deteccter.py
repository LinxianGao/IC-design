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
- 多模态感知融合（目标检测 + 盲道分割 + 交通标志OCR）
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
        self.bmodel_path = "/data/dataset/bmodels/ten_agu_f32.bmodel"
        self.blind_road_bmodel_path = "/data/dataset/bmodels/blind_road_f32.bmodel"
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
        
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('BlindAssistant')
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
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
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0


# ==================== 盲道检测器类 ====================
class BlindRoadDetector:
    """盲道检测器"""
    
    def __init__(self, bmodel_path, conf_thresh=0.3):
        self.conf_thresh = conf_thresh
        self.engine = sail.Engine(bmodel_path, 0, sail.IOMode.SYSIO)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.output_names = self.engine.get_output_names(self.graph_name)
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def detect(self, img):
        """检测盲道"""
        h_img, w_img = img.shape[:2]
        
        # 预处理
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        
        # 推理
        outputs = self.engine.process(self.graph_name, {self.input_name: img_input})
        
        if 'output0_Concat' in outputs:
            out = outputs['output0_Concat']
            detections = []
            
            for i in range(out.shape[2]):
                cx = out[0, 0, i]
                cy = out[0, 1, i]
                bw = out[0, 2, i]
                bh = out[0, 3, i]
                conf = self.sigmoid(out[0, 4, i])
                
                if conf < self.conf_thresh:
                    continue
                
                x1 = (cx - bw/2) * w_img
                y1 = (cy - bh/2) * h_img
                x2 = (cx + bw/2) * w_img
                y2 = (cy + bh/2) * h_img
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })
            
            return detections
        
        return []
    
    def get_nearest(self, img):
        """获取最近盲道信息"""
        detections = self.detect(img)
        if detections:
            nearest = min(detections, key=lambda x: x['bbox'][3] - x['bbox'][1])
            h_img = img.shape[0]
            center_y = (nearest['bbox'][1] + nearest['bbox'][3]) // 2
            distance = (h_img - center_y) / h_img * 5  # 粗略估算
            
            return {
                'detected': True,
                'direction': '正前方',
                'distance': round(distance, 1),
                'bbox': nearest['bbox'],
                'confidence': nearest['confidence']
            }
        
        return {'detected': False, 'direction': None, 'distance': None}


# ==================== 交通标志识别类（OCR） ====================
class TrafficSignRecognizer:
    """交通标志文字识别器"""
    
    def __init__(self, use_ocr=True):
        self.use_ocr = use_ocr
        if use_ocr:
            try:
                import easyocr
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                self.available = True
            except ImportError:
                print("easyocr 未安装，交通标志文字识别功能禁用")
                self.available = False
        else:
            self.available = False
    
    def recognize(self, img, bbox):
        """识别交通标志中的文字"""
        if not self.available:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            
            results = self.reader.readtext(roi)
            if results:
                text = ' '.join([r[1] for r in results])
                confidence = sum([r[2] for r in results]) / len(results)
                return {'text': text, 'confidence': round(confidence, 2)}
        except Exception as e:
            print(f"OCR识别失败: {e}")
        
        return None


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
        self.num_classes = None
        
        self.logger.info(f"输入形状: {self.input_shape}")
        self.logger.info(f"输出名称: {self.output_names}")
        print(f"输入形状: {self.input_shape}")
        print(f"输出名称: {self.output_names}")
        
        # 类别名称
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
        
        # 初始化盲道检测器
        self.blind_road_detector = None
        if os.path.exists(self.config.blind_road_bmodel_path):
            try:
                self.blind_road_detector = BlindRoadDetector(self.config.blind_road_bmodel_path)
                self.logger.info("盲道检测器初始化成功")
            except Exception as e:
                self.logger.warning(f"盲道检测器初始化失败: {e}")
        
        # 初始化交通标志识别器
        self.sign_recognizer = TrafficSignRecognizer(use_ocr=True)
    
    # ==================== 基础函数 ====================
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_fps(self):
        return self.fps_monitor.get_fps()
    
    def update_fps(self):
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
        
        if depth_map is not None:
            try:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                h, w = depth_map.shape[:2]
                center_x = max(0, min(w-1, center_x))
                center_y = max(0, min(h-1, center_y))
                distance = depth_map[center_y, center_x]
                if distance > 0:
                    return round(float(distance), 1)
            except Exception as e:
                self.logger.warning(f"深度图距离估算失败: {e}")
        
        x1, y1, x2, y2 = bbox
        box_height = y2 - y1
        if box_height > 0:
            distance = (object_height * focal_length) / box_height
            return round(distance, 1)
        
        return -1
    
    # ==================== 盲道检测函数 ====================
    def detect_blind_road(self, img):
        """检测盲道"""
        if self.blind_road_detector:
            return self.blind_road_detector.get_nearest(img)
        return {'detected': False, 'direction': None, 'distance': None}
    
    # ==================== 交通标志文字识别 ====================
    def recognize_sign_text(self, img, bbox):
        """识别交通标志文字"""
        if self.sign_recognizer.available:
            return self.sign_recognizer.recognize(img, bbox)
        return None
    
    # ==================== 解码函数 ====================
    def decode_yolov8(self, output, img_shape):
        """解码 YOLOv8 输出"""
        h_img, w_img = img_shape[:2]
        _, feat_dim, num_boxes = output.shape
        self.num_classes = feat_dim - 4
        
        detections = []
        
        for i in range(num_boxes):
            cx = output[0, 0, i]
            cy = output[0, 1, i]
            bw = output[0, 2, i]
            bh = output[0, 3, i]
            conf = self.sigmoid(output[0, 4, i])
            
            if conf < self.conf_thresh:
                continue
            
            cls_scores = output[0, 5:self.num_classes+5, i]
            cls_id = np.argmax(cls_scores)
            cls_conf = self.sigmoid(cls_scores[cls_id])
            score = conf * cls_conf
            
            if score < self.conf_thresh:
                continue
            
            x1 = (cx - bw / 2) * w_img
            y1 = (cy - bh / 2) * h_img
            x2 = (cx + bw / 2) * w_img
            y2 = (cy + bh / 2) * h_img
            
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_img, int(x2))
            y2 = min(h_img, int(y2))
            
            if x2 > x1 and y2 > y1:
                detections.append([x1, y1, x2, y2, cls_id, score])
        
        return detections
    
    def decode_yolov5(self, outputs, img_shape):
        """解码 YOLOv5 输出"""
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
                        conf = self.sigmoid(out[0, a, i, j, 4])
                        if conf < self.conf_thresh:
                            continue
                        
                        cls_scores = out[0, a, i, j, 5:]
                        cls_id = np.argmax(cls_scores)
                        cls_conf = self.sigmoid(cls_scores[cls_id])
                        score = conf * cls_conf
                        
                        if score < self.conf_thresh:
                            continue
                        
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
    
    # ==================== 核心检测函数 ====================
    def detect(self, img, with_info=False):
        """
        检测图像中的目标
        
        Args:
            img: OpenCV 图像 (BGR格式)
            with_info: 是否返回方向、距离等信息
        
        Returns:
            detections: 检测结果列表
        """
        fps = self.update_fps()
        img_input = self.preprocess(img)
        outputs = self.engine.process(self.graph_name, {self.input_name: img_input})
        
        if 'output0_Concat' in outputs:
            detections = self.decode_yolov8(outputs['output0_Concat'], img.shape)
        else:
            detections = self.decode_yolov5(outputs, img.shape)
        
        detections = self.nms(detections)
        
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
    
    # ==================== 多模态感知融合 ====================
    def perceive(self, img, with_ocr=True):
        """
        多模态感知：融合目标检测、盲道检测、交通标志识别
        
        Args:
            img: OpenCV 图像
            with_ocr: 是否进行OCR识别
        
        Returns:
            perception_result: 综合感知结果字典
        """
        # 初始化结果
        perception_result = {
            "direction": None,
            "distance": None,
            "obs_type": None,
            "obs_on_blind_road": False,
            "blind_road_detected": False,
            "blind_road_direction": None,
            "blind_road_distance": None,
            "sign_content": None,
            "sign_distance": None,
            "sign_direction": None
        }
        
        # 1. 目标检测
        detections = self.detect(img, with_info=True)
        
        if detections:
            # 找出最近的障碍物
            nearest = min(detections, key=lambda x: x['distance'])
            perception_result["direction"] = nearest['direction']
            perception_result["distance"] = nearest['distance']
            perception_result["obs_type"] = nearest['class_name_zh']
            
            # 2. 检查障碍物是否在盲道上（需要盲道信息）
            blind_road_info = self.detect_blind_road(img)
            if blind_road_info['detected']:
                # 简化判断：检查障碍物是否在盲道区域附近
                obs_center = (nearest['bbox'][0] + nearest['bbox'][2]) // 2
                road_center = (blind_road_info['bbox'][0] + blind_road_info['bbox'][2]) // 2 if 'bbox' in blind_road_info else None
                if road_center and abs(obs_center - road_center) < 100:
                    perception_result["obs_on_blind_road"] = True
        
        # 3. 盲道检测
        blind_road_info = self.detect_blind_road(img)
        if blind_road_info['detected']:
            perception_result["blind_road_detected"] = True
            perception_result["blind_road_direction"] = blind_road_info['direction']
            perception_result["blind_road_distance"] = blind_road_info['distance']
        
        # 4. 交通标志识别（仅对施工标志等交通标志类进行OCR）
        if with_ocr:
            for det in detections:
                # 只对施工标志进行OCR识别
                if det['class_id'] in [0, 2, 3]:  # 施工标志、禁止通行、公交站牌
                    sign_text = self.recognize_sign_text(img, det['bbox'])
                    if sign_text:
                        perception_result["sign_content"] = sign_text['text']
                        perception_result["sign_distance"] = det['distance']
                        perception_result["sign_direction"] = det['direction']
                        break  # 只取第一个识别的标志
        
        return perception_result, detections
    
    # ==================== 辅助函数 ====================
    def get_class_name(self, class_id, lang='en'):
        if lang == 'zh':
            return self.class_names_zh.get(class_id, f'class_{class_id}')
        return self.class_names.get(class_id, f'class_{class_id}')
    
    def set_conf_thresh(self, thresh):
        self.conf_thresh = thresh
        self.logger.info(f"置信度阈值设置为: {thresh}")
    
    def set_nms_thresh(self, thresh):
        self.nms_thresh = thresh
        self.logger.info(f"NMS 阈值设置为: {thresh}")
    
    # ==================== 可视化函数 ====================
    def draw_detections(self, img, detections, show_info=True):
        """在图像上绘制检测结果"""
        colors = {
            0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0),
            3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255),
            6: (128, 128, 0), 7: (0, 128, 128), 8: (128, 0, 128), 9: (128, 128, 128)
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            name = det.get('class_name_zh', det.get('class_name', 'unknown'))
            
            color = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if show_info and 'direction' in det and 'distance' in det:
                label = f"{name}: {conf:.2f} | {det['direction']} {det['distance']:.1f}m"
            else:
                label = f"{name}: {conf:.2f}"
            
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        fps = self.get_fps()
        cv2.putText(img, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    
    def draw_perception(self, img, perception_result):
        """绘制感知结果"""
        h, w = img.shape[:2]
        
        # 显示盲道信息
        if perception_result['blind_road_detected']:
            cv2.putText(img, f"盲道: {perception_result['blind_road_direction']} {perception_result['blind_road_distance']:.1f}m",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示障碍物信息
        if perception_result['obs_type']:
            obs_text = f"障碍: {perception_result['obs_type']} {perception_result['direction']} {perception_result['distance']:.1f}m"
            if perception_result['obs_on_blind_road']:
                obs_text += " ⚠️在盲道上"
            cv2.putText(img, obs_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示交通标志信息
        if perception_result['sign_content']:
            cv2.putText(img, f"标志: {perception_result['sign_content']} {perception_result['sign_direction']} {perception_result['sign_distance']:.1f}m",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return img


# ==================== 多模型检测类 ====================
class MultiModelDetector:
    """多模型检测器"""
    
    def __init__(self):
        self.detectors = {}
    
    def add_detector(self, name, detector):
        self.detectors[name] = detector
    
    def detect(self, img, with_info=False):
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
    print("=" * 50)
    print("YOLO Detector 测试")
    print("=" * 50)
    
    BMODEL_PATH = "/data/dataset/bmodels/ten_classes_f32.bmodel"
    TEST_IMAGE = "/data/dataset/runs/detect/盲道.webp"
    
    detector = YOLODetector(BMODEL_PATH, conf_thresh=0.5, nms_thresh=0.45)
    
    # 测试多模态感知
    print(f"\n多模态感知测试 - 图片: {TEST_IMAGE}")
    img = cv2.imread(TEST_IMAGE)
    if img is not None:
        perception_result, detections = detector.perceive(img, with_ocr=True)
        
        print("\n感知结果:")
        print(json.dumps(perception_result, ensure_ascii=False, indent=4))
        
        # 绘制结果
        result_img = detector.draw_detections(img.copy(), detections)
        result_img = detector.draw_perception(result_img, perception_result)
        cv2.imwrite('/tmp/perception_result.jpg', result_img)
        print("\n结果图片已保存到: /tmp/perception_result.jpg")
    
    # 测试摄像头
    print("\n尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            perception_result, detections = detector.perceive(frame)
            print(f"实时感知 - 障碍物: {perception_result['obs_type']}, 盲道检测: {perception_result['blind_road_detected']}")
        cap.release()
    else:
        print("摄像头不可用")
    
    print("\n测试完成！")
