# zero_dce.py - 夜间图像增强模块
# 运行位置：少林派开发板上
# 调用者：main.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from sophon.sail import Engine, Bmcv, Handle


class DCE_Net(nn.Module):
    """零参考深度曲线估计网络
    论文: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)
    特点: 极轻量 (约79K参数/8层)，不需要配对训练数据，推理极快[citation:3][citation:5]
    """
    
    def __init__(self, n_channels=3, n_iters=8):
        super(DCE_Net, self).__init__()
        
        # 特征提取层 (7层卷积)
        self.conv1 = self._make_conv_layer(n_channels, 32)
        self.conv2 = self._make_conv_layer(32, 32)
        self.conv3 = self._make_conv_layer(32, 32)
        self.conv4 = self._make_conv_layer(32, 32)
        self.conv5 = self._make_conv_layer(32, 32)
        self.conv6 = self._make_conv_layer(32, 32)
        self.conv7 = self._make_conv_layer(32, 32)
        
        # 输出层: 预测 RGB 三通道的 α 参数
        # 输出通道数 = n_channels * n_iters (每个通道每轮迭代一个α)
        self.conv8 = nn.Conv2d(32, n_channels * n_iters, kernel_size=3, stride=1, padding=1)
        
        self.n_iters = n_iters
        
        # 初始化权重 (论文中的默认初始化)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_conv_layer(self, in_ch, out_ch):
        """创建带 ReLU 激活的卷积层"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        输入: x - [B, 3, H, W]，范围0-1
        输出: α参数列表 - 长度为n_iters，每个shape为[B, 3, H, W]，范围-1到1
        """
        # 特征提取
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)  # 残差连接
        x4 = self.conv4(x3)
        x5 = self.conv5(x4 + x3)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6 + x5)
        
        # 预测所有迭代的α参数
        alphas = self.conv8(x7)  # [B, 3*n_iters, H, W]
        alphas = torch.tanh(alphas)  # 限制到 [-1, 1]
        
        # 按迭代次数拆分
        alphas_split = torch.split(alphas, split_size_or_sections=3, dim=1)
        
        return alphas_split


class LightEnhanceCurve:
    """光照增强曲线计算器
    公式: LE(I) = I + α * I * (1 - I)
    特点: 保证输出在[0,1]区间内，保持相邻像素的单调性，可微[citation:3][citation:5]
    """
    
    @staticmethod
    def apply_curve(x, alpha):
        """
        单次曲线应用
        x: 输入图像，范围[0,1]
        alpha: 曲线参数，范围[-1,1]
        返回: 增强后图像，范围[0,1]
        """
        return x + alpha * x * (1 - x)
    
    @staticmethod
    def apply_iterative(x, alphas):
        """
        迭代应用曲线
        x: 输入图像，[B, C, H, W]
        alphas: 每轮的参数列表
        返回: 增强后图像
        """
        for alpha in alphas:
            x = LightEnhanceCurve.apply_curve(x, alpha)
        return x


class ZeroDCEPostTraining:
    """Zero-DCE的损失函数集合
    这些损失不需要配对数据，通过图像本身的特性引导学习[citation:5]
    """
    
    @staticmethod
    def spatial_consistency_loss(original, enhanced):
        """
        空间一致性损失: 保持增强图和原图相邻区域的相对差异
        公式: 计算相邻像素差值的L1损失
        """
        # 计算水平和垂直方向的梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(original.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(original.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        grad_orig_x = F.conv2d(original, sobel_x, padding=1, groups=3)
        grad_orig_y = F.conv2d(original, sobel_y, padding=1, groups=3)
        grad_enh_x = F.conv2d(enhanced, sobel_x, padding=1, groups=3)
        grad_enh_y = F.conv2d(enhanced, sobel_y, padding=1, groups=3)
        
        loss = F.l1_loss(grad_enh_x, grad_orig_x) + F.l1_loss(grad_enh_y, grad_orig_y)
        return loss
    
    @staticmethod
    def exposure_control_loss(image, patch_size=16, target_mean=0.6):
        """
        曝光控制损失: 将图像划分为多个patch，每个patch的平均亮度接近目标值0.6
        论文中建议目标均值为0.6[citation:5]
        """
        B, C, H, W = image.shape
        # 将图像划分为patch
        patch_h = H // patch_size
        patch_w = W // patch_size
        
        # 计算每个patch的均值
        patches = image[:, :, :patch_h*patch_size, :patch_w*patch_size]
        patches = patches.view(B, C, patch_h, patch_size, patch_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(-1, C, patch_size, patch_size)
        
        mean = patches.mean(dim=[2, 3])
        loss = torch.mean((mean - target_mean) ** 2)
        return loss
    
    @staticmethod
    def color_constancy_loss(image):
        """
        颜色恒常性损失: 确保增强后RGB三通道的平均值相近，防止偏色
        """
        B, C, H, W = image.shape
        mean_rgb = image.view(B, C, -1).mean(dim=2)  # [B, 3]
        
        # 计算各通道均值之间的差异
        diff_rg = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 1]).mean()
        diff_rb = torch.abs(mean_rgb[:, 0] - mean_rgb[:, 2]).mean()
        diff_gb = torch.abs(mean_rgb[:, 1] - mean_rgb[:, 2]).mean()
        
        return diff_rg + diff_rb + diff_gb
    
    @staticmethod
    def illumination_smoothness_loss(alphas):
        """
        光照平滑度损失: 保证曲线参数图的空间平滑性
        """
        loss = 0
        for alpha in alphas:
            # 计算水平和垂直方向的梯度
            grad_h = torch.abs(alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).mean()
            grad_w = torch.abs(alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).mean()
            loss += grad_h + grad_w
        return loss.mean()


class ImageEnhancer:
    """
    零参考深度曲线估计图像增强器
    适用于低光照、夜间场景的图像增强
    
    工作流程:
    1. DCE-Net预测每个像素的曲线参数
    2. 将参数应用到输入图像上，迭代8次得到增强图
    3. 整个过程仅需约79K参数，推理极快[citation:3][citation:5]
    """
    
    def __init__(self, model_path="models/zero_dce_int8.bmodel", use_npu=True):
        """
        初始化增强器
        参数:
            model_path: BModel模型路径
            use_npu: 是否使用NPU（False则使用PyTorch推理）
        """
        self.use_npu = use_npu
        self.n_iters = 8
        
        if use_npu:
            # 使用SAIL加载NPU模型
            self.handle = Handle(0)
            self.engine = Engine(model_path)
            self.input_shape = self.engine.get_input_shape(0)
            self.bmcv = Bmcv(self.handle)
        else:
            # 开发测试时使用PyTorch
            self.model = DCE_Net(n_channels=3, n_iters=self.n_iters)
            self.model.eval()
        
        print(f"[ZeroDCE] 初始化完成, NPU模式: {use_npu}")
    
    def enhance_pytorch(self, image: np.ndarray) -> np.ndarray:
        """
        使用PyTorch进行增强（开发测试用）
        输入: BGR图像 [H,W,3], uint8
        返回: 增强后BGR图像, uint8
        """
        # BGR to RGB + 归一化
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        
        with torch.no_grad():
            # 预测曲线参数
            alphas = self.model(img_tensor)
            # 应用曲线增强
            enhanced = LightEnhanceCurve.apply_iterative(img_tensor, alphas)
        
        # 转回numpy
        enhanced = enhanced.squeeze(0).permute(1, 2, 0).numpy()
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        # 转回BGR格式
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        return enhanced
    
    def enhance_npu(self, image: np.ndarray) -> np.ndarray:
        """
        使用少林派NPU进行增强
        输入: BGR图像 [H,W,3], uint8
        返回: 增强后BGR图像, uint8
        """
        # 预处理: BGR to RGB, resize到模型输入尺寸, 归一化
        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_shape[3], self.input_shape[2]))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_chw = img_normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)
        
        # NPU推理
        outputs = self.engine.process([input_tensor])
        
        # 后处理: 根据模型输出重建增强图像
        # 注意: BModel的输出格式取决于转换方式
        # 这里假设输出是增强后的图像
        enhanced = outputs[0][0]  # [3, H, W]
        enhanced = enhanced.transpose(1, 2, 0)  # [H, W, 3]
        
        # 如果输出尺寸和原图不同，需要resize回去
        if enhanced.shape[0] != h or enhanced.shape[1] != w:
            enhanced = cv2.resize(enhanced, (w, h))
        
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """统一的增强接口"""
        if self.use_npu:
            return self.enhance_npu(image)
        else:
            return self.enhance_pytorch(image)
    
    def enhance_if_needed(self, image: np.ndarray, is_night: bool = True) -> np.ndarray:
        """
        条件增强: 只在夜间场景下增强
        """
        if is_night:
            return self.enhance(image)
        return image


# ========== 工具函数 ==========

def is_night_scene(image: np.ndarray, threshold: int = 80) -> bool:
    """
    判断图像是否为夜间场景
    基于图像平均亮度
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """图像预处理"""
    if len(image.shape) == 2:  # 灰度图转3通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.astype(np.float32) / 255.0


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 测试代码
    enhancer = ImageEnhancer(use_npu=False)
    
    # 读取夜间图像
    img = cv2.imread("night_shot.jpg")
    if img is not None:
        # 增强
        enhanced = enhancer.enhance(img)
        
        # 显示对比
        cv2.imshow("Original", img)
        cv2.imshow("Enhanced", enhanced)
        cv2.waitKey(0)
