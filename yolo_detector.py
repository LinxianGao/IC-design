# yolo_detector.py - 完整实现
import sophon.sail as sail
import cv2
import numpy as np
import time
from typing import List, Dict, Tuple

# COCO数据集类别名称（YOLOv5/v8默认）
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class YOLODetector:
    def __init__(self, bmodel_path: str, conf_threshold: float = 0.5):
        """
        初始化YOLO检测器
        
        Args:
            bmodel_path: BModel文件路径（如 yolov5s_1684x_int8.bmodel）
            conf_threshold: 置信度阈值，默认0.5
        """
        self.conf_threshold = conf_threshold
        
        # 1. 初始化设备句柄（BM1684X的第一块TPU）[citation:6]
        self.handle = sail.BMHandle()
        ret = self.handle.init(0)
        if ret != 0:
            raise RuntimeError("TPU设备初始化失败，请检查硬件连接")
        
        # 2. 加载BModel
        self.model = sail.BMModel(self.handle, bmodel_path)
        
        # 3. 获取模型输入输出信息
        self.input_shape = self.model.get_input_shape(0)  # 如 [1, 3, 640, 640]
        self.output_shape = self.model.get_output_shape(0)  # 如 [1, 1, 200, 7]
        
        # 4. 获取输入尺寸
        self.input_height = self.input_shape[2]  # 640
        self.input_width = self.input_shape[3]   # 640
        
        # 5. 预分配输入Tensor（避免重复分配，提升性能）
        self.input_tensor = sail.BMData(self.handle, self.input_shape)
        
        print(f"[YOLODetector] 初始化成功")
        print(f"  - 模型路径: {bmodel_path}")
        print(f"  - 输入尺寸: {self.input_width}x{self.input_height}")
        print(f"  - 置信度阈值: {self.conf_threshold}")
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理：将输入图像resize到模型输入尺寸
        
        Args:
            frame: 原始BGR图像
            
        Returns:
            resize后的图像 (height, width, 3)
        """
        # resize到模型输入尺寸（640x640）
        img_resized = cv2.resize(frame, (self.input_width, self.input_height))
        return img_resized
    
    def _postprocess(self, outputs, original_h: int, original_w: int) -> List[Dict]:
        """
        后处理：解析模型输出，转换坐标到原图尺寸
        
        模型输出shape: [1, 1, 200, 7]
        其中200表示最大检测框数，7个值含义：
        [batch_id, class_id, score, center_x, center_y, width, height]
        
        注意：center_x, center_y, width, height都是相对于640x640的归一化坐标
        
        Args:
            outputs: 模型原始输出
            original_h: 原图高度
            original_w: 原图宽度
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 将输出转换为numpy数组
        detections = outputs[0].to_numpy()  # shape: (1, 1, 200, 7)
        
        # 遍历所有检测框
        for i in range(detections.shape[2]):
            box = detections[0, 0, i]
            batch_id = int(box[0])
            class_id = int(box[1])
            score = float(box[2])
            
            # 过滤低置信度
            if score < self.conf_threshold:
                continue
            
            # 归一化坐标（相对于640x640）
            cx = box[3]  # 中心点x (0-1)
            cy = box[4]  # 中心点y (0-1)
            w_norm = box[5]  # 宽度 (0-1)
            h_norm = box[6]  # 高度 (0-1)
            
            # 转换为绝对像素坐标（相对于原图）
            x1 = int((cx - w_norm / 2) * original_w)
            y1 = int((cy - h_norm / 2) * original_h)
            x2 = int((cx + w_norm / 2) * original_w)
            y2 = int((cy + h_norm / 2) * original_h)
            
            # 边界裁剪
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            # 获取类别名称
            class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"
            
            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'score': score,
                'bbox': [x1, y1, x2, y2]
            })
        
        return results
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测图像中的目标
        
        Args:
            frame: BGR图像，numpy数组
            
        Returns:
            检测结果列表
        """
        if frame is None or frame.size == 0:
            return []
        
        original_h, original_w = frame.shape[:2]
        
        # 1. 预处理
        img_resized = self._preprocess(frame)
        
        # 2. 将图像数据拷贝到输入Tensor
        self.input_tensor.from_numpy(img_resized.astype(np.uint8))
        
        # 3. 执行推理
        outputs = self.model.process([self.input_tensor])
        
        # 4. 后处理
        results = self._postprocess(outputs, original_h, original_w)
        
        return results
    
    def detect_with_time(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        检测图像中的目标，同时返回推理耗时
        
        Args:
            frame: BGR图像
            
        Returns:
            (检测结果列表, 推理耗时(毫秒))
        """
        start_time = time.time()
        results = self.detect(frame)
        elapsed_ms = (time.time() - start_time) * 1000
        return results, elapsed_ms
    
    def get_fps(self, frame: np.ndarray, iterations: int = 100) -> float:
        """
        测试推理FPS
        
        Args:
            frame: 测试图像
            iterations: 测试次数
            
        Returns:
            平均FPS
        """
        total_time = 0
        for _ in range(iterations):
            _, elapsed_ms = self.detect_with_time(frame)
            total_time += elapsed_ms
        avg_ms = total_time / iterations
        return 1000 / avg_ms
