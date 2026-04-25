#!/usr/bin/env python3
"""
OCR 中文路牌识别器封装
用于识别路牌上的文字（如"限速40"、"前方学校"等）
"""

import cv2
import numpy as np

class OCRRecognizer:
    def __init__(self, use_gpu=False, languages=['ch_sim', 'en']):
        """
        初始化 OCR 识别器
        
        参数:
            use_gpu: 是否使用 GPU（少林派上通常设为 False）
            languages: 识别语言列表，'ch_sim'=简体中文，'en'=英文
        """
        self.use_gpu = use_gpu
        self.languages = languages
        
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
            print(f"[OCR] 初始化成功，语言: {languages}")
        except Exception as e:
            print(f"[OCR] 初始化失败: {e}")
            self.reader = None
    
    def recognize_from_bbox(self, image, bbox, min_confidence=0.5):
        """
        从单个检测框区域识别文字
        
        参数:
            image: 原始图像 (numpy array, BGR格式)
            bbox: 检测框 [x1, y1, x2, y2]
            min_confidence: 最低置信度阈值
            
        返回:
            识别到的文字，如果没有识别到返回 None
        """
        if self.reader is None:
            return None
        
        # 裁剪检测框区域
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # 转换为 RGB（easyocr 需要 RGB 格式）
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # 识别
        try:
            results = self.reader.readtext(roi_rgb)
            if results:
                # 取置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]
                if confidence >= min_confidence:
                    return text
        except Exception as e:
            print(f"[OCR] 识别失败: {e}")
        
        return None
    
    def recognize_multi(self, image, detections, min_confidence=0.5):
        """
        从多个检测框区域识别文字
        
        参数:
            image: 原始图像 (numpy array, BGR格式)
            detections: YOLO 检测结果列表，每个元素包含 'bbox' 和 'class_name'
            min_confidence: 最低置信度阈值
            
        返回:
            识别结果列表，每个元素包含 bbox, text, confidence
        """
        if self.reader is None:
            return []
        
        results = []
        
        for det in detections:
            # 只处理路牌类别的检测
            class_name = det.get('class_name', '').lower()
            if not any(keyword in class_name for keyword in ['sign', 'traffic', 'stop', 'speed']):
                continue
            
            bbox = det.get('bbox')
            if bbox is None:
                continue
            
            text = self.recognize_from_bbox(image, bbox, min_confidence)
            if text:
                results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': det.get('confidence', 0),
                    'original_class': class_name
                })
        
        return results
    
    def recognize_from_roi(self, roi_image):
        """
        直接从 ROI 图像区域识别文字
        
        参数:
            roi_image: 裁剪后的图像区域 (numpy array, BGR格式)
            
        返回:
            识别到的文字，如果没有识别到返回 None
        """
        if self.reader is None:
            return None
        
        if roi_image.size == 0:
            return None
        
        # 转换为 RGB
        roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.reader.readtext(roi_rgb)
            if results:
                best_result = max(results, key=lambda x: x[2])
                return best_result[1]
        except Exception as e:
            print(f"[OCR] 识别失败: {e}")
        
        return None


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.append('/data')
    from camera import Camera
    
    print("=" * 50)
    print("OCR 识别器测试")
    print("=" * 50)
    
    # 初始化 OCR
    ocr = OCRRecognizer(use_gpu=False)
    
    # 初始化摄像头
    cam = Camera()
    
    # 获取一帧图像
    frame = cam.get_frame()
    cam.release()
    
    if frame is not None:
        # 测试单张图片识别（假设有路牌区域）
        print("\n【测试】如果图片中有路牌，可以手动指定 bbox 进行测试")
        
        # 示例：手动指定一个区域进行识别
        # 你需要根据实际图像内容调整坐标
        # text = ocr.recognize_from_bbox(frame, [100, 100, 300, 200])
        # print(f"识别结果: {text}")
        
        print("\n提示：此测试需要手动调整 bbox 坐标")
    else:
        print("无法获取图像")
