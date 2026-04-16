"""
YOLO检测器统一入口
通过环境变量 USE_REMOTE 控制使用本地模型还是远程模型
"""

import os
import cv2

# 检查是否使用远程模式
USE_REMOTE = os.environ.get('USE_REMOTE', 'False').lower() == 'true'


class YOLODetector:
    """
    统一的YOLO检测器类
    根据 USE_REMOTE 自动选择本地PyTorch版本或远程少林派版本
    """
    
    def __init__(self, model_path=None):
        """
        初始化检测器
        
        参数:
            model_path: 模型路径（本地模式时可选，远程模式时由A提供）
        """
        self.use_remote = USE_REMOTE
        
        if self.use_remote:
            print("[检测器] 使用远程模式（少林派）")
            self._init_remote(model_path)
        else:
            print("[检测器] 使用本地模式（PyTorch）")
            self._init_local(model_path)
    
    def _init_local(self, model_path):
        """初始化本地PyTorch YOLO模型"""
        import torch
        
        print("  正在加载本地YOLO模型...")
        
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                        path=model_path, force_reload=False)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                        pretrained=True, force_reload=False)
        
        self.model.conf = 0.5
        self.valid_classes = ['person', 'car', 'truck', 'bus', 
                              'bicycle', 'motorcycle']
        
        print("  本地YOLO模型加载完成！")
    
    def _init_remote(self, model_path):
        """
        初始化远程少林派模型
        
        ============================================================
        以下代码是从A给的 detector_remote.py 复制过来的
        ============================================================
        """
        import numpy as np
        from sophon.sail import Engine  # 少林派的AI推理库
        
        print("  正在加载少林派YOLO模型...")
        
        # 如果没有指定模型路径，使用默认值
        if model_path is None:
            model_path = "yolov5s_int8.bmodel"
        
        self.engine = Engine(model_path)
        self.input_shape = (640, 640)
        self.conf_threshold = 0.5
        
        # 类别映射表
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        
        print(f"  少林派YOLO模型加载完成！模型路径: {model_path}")
    
    def _init_remote_old(self, model_path):
        """
        ⚠️ 这个函数可以删掉，上面的 _init_remote 已经替换好了
        保留在这里只是作为对比
        """
        pass
    
    def estimate_distance(self, box_height, frame_height=480):
        """根据检测框高度估算距离（米）"""
        REAL_HEIGHT = 1.7
        FOCAL_LENGTH = 500
        
        if box_height <= 0:
            return 99
        
        distance = (REAL_HEIGHT * FOCAL_LENGTH) / box_height
        return round(distance, 2)
    
    def detect(self, frame):
        """检测图片中的障碍物"""
        if self.use_remote:
            return self._detect_remote(frame)
        else:
            return self._detect_local(frame)
    
    def _detect_local(self, frame):
        """本地PyTorch版本的检测实现"""
        results = self.model(frame)
        
        detections = []
        
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            label = self.model.names[int(cls)]
            
            if label not in self.valid_classes:
                continue
            
            box_height = y2 - y1
            distance = self.estimate_distance(box_height, frame.shape[0])
            
            detections.append({
                'label': label,
                'confidence': round(conf, 2),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'distance': distance
            })
        
        return detections
    
    def _detect_remote(self, frame):
        """
        远程少林派版本的检测实现
        
        ============================================================
        以下代码是从A给的 detector_remote.py 复制过来的
        ============================================================
        """
        import numpy as np
        
        # 1. 预处理：resize到640x640
        img_resized = cv2.resize(frame, self.input_shape)
        
        # 2. 转换格式：HWC -> CHW，归一化
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))  # CHW格式
        
        # 3. 推理
        outputs = self.engine.infer([img_input])
        
        # 4. 解析输出（假设输出格式是[1, 1, 200, 7]）
        detections = []
        output_data = outputs[0].reshape(-1, 7)
        
        for i in range(output_data.shape[0]):
            batch_id, class_id, score, cx, cy, w, h = output_data[i]
            
            if score < self.conf_threshold:
                continue
            
            # 将归一化坐标转换回原图尺寸
            x1 = int((cx - w/2) * frame.shape[1])
            y1 = int((cy - h/2) * frame.shape[0])
            x2 = int((cx + w/2) * frame.shape[1])
            y2 = int((cy + h/2) * frame.shape[0])
            
            # 获取类别名称
            class_id_int = int(class_id)
            if class_id_int < len(self.classes):
                label = self.classes[class_id_int]
            else:
                label = 'unknown'
            
            # 估算距离
            box_height = y2 - y1
            distance = self.estimate_distance(box_height, frame.shape[0])
            
            detections.append({
                'label': label,
                'confidence': float(score),
                'bbox': [x1, y1, x2, y2],
                'distance': distance
            })
        
        return detections
    
    def detect_and_draw(self, frame):
        """检测并在图片上画框（用于显示）"""
        detections = self.detect(frame)
        result_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            distance = det['distance']
            
            # 根据距离选择颜色
            if distance < 1.0:
                color = (0, 0, 255)      # 红色：很近
            elif distance < 2.0:
                color = (0, 165, 255)    # 橙色：较近
            else:
                color = (0, 255, 0)      # 绿色：安全
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {distance}m ({conf:.0%})"
            cv2.putText(result_frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame, detections


# 测试代码
if __name__ == "__main__":
    print("="*50)
    print("测试YOLO检测器统一入口")
    print("="*50)
    
    # 测试本地模式
    print("\n1. 测试本地模式...")
    detector = YOLODetector()
    print("   ✅ 本地模式初始化成功")
    
    # 测试远程模式（需要真实硬件）
    print("\n2. 测试远程模式...")
    print("   设置 USE_REMOTE=True 后使用远程模式")
    print("   ⚠️ 需要少林派硬件和A提供的模型文件")
    
    print("\n✅ 测试完成")