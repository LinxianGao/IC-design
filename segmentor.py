# segmentor.py - A负责编写
# 运行位置：少林派开发板NPU上
# 调用者：main.py（C写的）

import numpy as np
import cv2
from sophon.sail import Engine, Bmcv, Handle

class Segmentor:
    """盲道语义分割类
    功能：将输入图像中的盲道区域分割出来，输出二值掩码
    """
    
    # ========== 函数1：初始化 ==========
    def __init__(self, model_path="models/pidnet_int8.bmodel"):
        """
        作用：加载分割模型到NPU
        输入：模型文件路径
        输出：无
        """
        # 创建设备句柄
        self.handle = Handle(0)
        
        # 加载BModel模型
        self.engine = Engine(model_path)
        
        # 获取模型输入尺寸
        self.input_shape = self.engine.get_input_shape(0)  # (1, 3, H, W)
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        # 初始化硬件图像处理工具
        self.bmcv = Bmcv(self.handle)
        
        # 类别配置（根据你的数据集调整）
        self.num_classes = 1  # 只分割盲道
        self.class_names = ['blind_road']
        
        print(f"[Segmentor] 模型加载成功，输入尺寸: {self.input_h}x{self.input_w}")
    
    # ========== 函数2：主分割函数（C会调用这个） ==========
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        作用：识别图像中的盲道区域
        输入：frame - BGR图像，shape=(H,W,3)，dtype=uint8
        返回：mask - 盲道掩码，shape=(H,W)，dtype=uint8，255=盲道，0=背景
        """
        if frame is None or frame.size == 0:
            return np.zeros((480, 640), dtype=np.uint8)
        
        original_h, original_w = frame.shape[:2]
        
        # 1. 预处理
        input_tensor = self._preprocess(frame)
        
        # 2. NPU推理
        outputs = self.engine.process([input_tensor])
        
        # 3. 后处理
        mask = self._postprocess(outputs, (original_h, original_w))
        
        return mask
    
    # ========== 函数3：预处理（内部使用） ==========
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        作用：将输入图像转换为模型需要的格式
        输入：frame - BGR图像 (H,W,3)
        返回：input_tensor (1,3,H_input,W_input)，float32，范围0-1
        """
        # 方法A：使用硬件加速预处理（推荐，更快）
        # 将numpy图像转换为BMImage格式（零拷贝）
        bmimg = BMImage(self.handle, frame.shape[1], frame.shape[0],
                        Format.FORMAT_BGR_PLANAR, ImgDtype.DATA_TYPE_EXT_1N_BYTE)
        bmimg.from_numpy(frame)
        
        # 硬件加速resize到模型输入尺寸
        resized = BMImage(self.handle, self.input_w, self.input_h,
                          Format.FORMAT_BGR_PLANAR, ImgDtype.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.resize(bmimg, resized)
        
        # 转为numpy并归一化
        img_np = resized.asnumpy()  # (H, W, 3)
        
        # 方法B：软件预处理（备选，如果没有硬件加速驱动）
        # img_np = cv2.resize(frame, (self.input_w, self.input_h))
        
        # BGR to RGB + 归一化 + HWC to CHW + 加batch维度
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = img_normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)
        
        return input_tensor
    
    # ========== 函数4：后处理（内部使用） ==========
    def _postprocess(self, outputs: list, original_shape: tuple) -> np.ndarray:
        """
        作用：解析分割结果，生成盲道掩码
        输入：
            outputs - 模型原始输出列表
            original_shape - 原图尺寸 (H, W)
        返回：mask - 二值掩码 (H, W)，dtype=uint8，255=盲道
        """
        # 模型输出格式取决于你的模型
        # 假设输出是 (1, num_classes, H_out, W_out) 的概率图
        
        # 获取分割概率图（单通道，因为只有盲道一类）
        if len(outputs[0].shape) == 4:
            prob_map = outputs[0][0, 0, :, :]  # (H_out, W_out)
        else:
            prob_map = outputs[0][0]  # (H_out, W_out)
        
        # 阈值化：概率>0.5的像素为盲道
        binary_map = (prob_map > 0.5).astype(np.uint8) * 255
        
        # 上采样回原图尺寸
        mask = cv2.resize(binary_map, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    # ========== 辅助函数：掩码与检测框匹配（C可能会用） ==========
    def is_on_blind_road(self, bbox: list, mask: np.ndarray, threshold: float = 0.3) -> bool:
        """
        作用：判断检测框是否在盲道上
        输入：
            bbox - [x1, y1, x2, y2]
            mask - 盲道掩码
            threshold - 重合比例阈值
        返回：True表示在盲道上
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        y1, y2 = max(0, y1), min(mask.shape[0], y2)
        x1, x2 = max(0, x1), min(mask.shape[1], x2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        roi = mask[y1:y2, x1:x2]
        blind_ratio = np.sum(roi > 0) / (roi.size + 1e-6)
        
        return blind_ratio > threshold
