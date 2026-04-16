"""
yolo_detector.py
少林派YOLO检测器 - 导盲场景专用
输出格式: {"direction": "左前方", "distance": 1.2, "type": "person", "scene": "normal"}
A角（板主）需要在开发板上运行此文件
"""

import cv2
import numpy as np
import json

# ========== 根据实际环境选择后端 ==========
try:
    from sophon.sail import BMHandle, BMModel, BMData
    USE_SAIL = True
    print("[INFO] 使用 sophon-sail 后端")
except ImportError:
    try:
        import onnxruntime as ort
        USE_SAIL = False
        print("[INFO] 使用 ONNX Runtime 后端（备选）")
    except ImportError:
        print("[ERROR] 请安装 sophon-sail 或 onnxruntime")
        exit(1)

# ========== 配置参数 ==========
CONF_THRESHOLD = 0.5          # 置信度阈值
INPUT_SIZE = 640              # 模型输入尺寸
FRAME_WIDTH = 640             # 图像宽度（用于方向判断）
FRAME_HEIGHT = 480            # 图像高度（用于距离估算）

# COCO 80个类别名称
CLASS_NAMES = [
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

# 导盲场景需要检测的类别（人、车、自行车、摩托车等）
TARGET_CLASSES = {'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'}

# 场景类型（可扩展）
SCENE_TYPE = "normal"  # normal, night, rain

# ========== 全局变量 ==========
_model = None
_handle = None


def _init_model(bmodel_path="yolov5s_1684x_f16.bmodel"):
    """初始化模型（只执行一次）"""
    global _model, _handle
    
    if USE_SAIL:
        _handle = BMHandle()
        _handle.init(0)
        bm_model = BMModel(_handle, bmodel_path)
        _model = bm_model.get_graph("yolov5s")
        print(f"[INFO] 模型加载成功: {bmodel_path}")
    else:
        _model = ort.InferenceSession(bmodel_path.replace('.bmodel', '.onnx'))
        print(f"[INFO] ONNX模型加载成功: {bmodel_path}")


def _preprocess(image):
    """
    预处理：resize + 归一化
    输入：BGR图像 (H, W, 3)
    输出：模型输入格式 (1, 3, 640, 640)
    """
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    input_tensor = np.expand_dims(chw, axis=0)
    return input_tensor


def _get_direction(center_x, frame_width=FRAME_WIDTH):
    """
    根据物体中心点x坐标判断方向
    返回: "左前方" / "正前方" / "右前方"
    """
    left_boundary = frame_width // 3
    right_boundary = 2 * frame_width // 3
    
    if center_x < left_boundary:
        return "左前方"
    elif center_x > right_boundary:
        return "右前方"
    else:
        return "正前方"


def _estimate_distance(box_height, frame_height=FRAME_HEIGHT):
    """
    根据边界框高度估算距离（米）
    公式：距离 = 参考高度(1米时) / 实际高度
    返回：距离（米），保留1位小数
    """
    if box_height <= 0:
        return 5.0
    
    # 假设：当物体高度占画面1/3时，距离为1米
    ref_height = frame_height / 3.0
    distance = ref_height / box_height
    
    # 限制距离范围 0.3 - 8.0 米
    distance = max(0.3, min(8.0, distance))
    return round(distance, 1)


def _postprocess(outputs, original_shape):
    """
    后处理：解析检测框，过滤低置信度
    输出格式：[{"direction": "左前方", "distance": 1.2, "type": "person", "scene": "normal"}, ...]
    """
    detections = outputs[0] if isinstance(outputs, list) else outputs
    
    # 解析输出格式
    if len(detections.shape) == 4 and detections.shape[2] == 200:
        boxes = detections[0, 0]  # (200, 7)
    elif len(detections.shape) == 3:
        boxes = detections[0]
    else:
        print("[WARNING] 模型输出格式异常，返回空列表")
        return []
    
    results = []
    h, w = original_shape[:2]
    
    for box in boxes:
        class_id = int(box[1])
        score = float(box[2])
        
        # 过滤低置信度
        if score < CONF_THRESHOLD:
            continue
        
        # 只保留导盲场景需要的类别
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
        if class_name not in TARGET_CLASSES:
            continue
        
        # 解析坐标
        cx, cy, bw, bh = box[3], box[4], box[5], box[6]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # 边界裁剪
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        center_x = (x1 + x2) // 2
        box_height = y2 - y1
        
        # 组装输出JSON格式
        results.append({
            "direction": _get_direction(center_x, w),
            "distance": _estimate_distance(box_height, h),
            "type": class_name,
            "scene": SCENE_TYPE
        })
    
    # 按距离排序（近的在前）
    results.sort(key=lambda x: x['distance'])
    return results


def _postprocess_raw(outputs, original_shape):
    """原始YOLO输出的后处理（备选方案）"""
    print("[WARNING] 使用原始后处理，建议转换时加上 --add_postprocess yolov5")
    return []


# ========== 对外暴露的核心函数 ==========

def detect_objects(image, conf_threshold=None, bmodel_path="yolov5s_1684x_f16.bmodel"):
    """
    检测图像中的障碍物
    
    参数:
        image: numpy数组，BGR格式 (H, W, 3)
        conf_threshold: 置信度阈值，默认0.5
        bmodel_path: 模型文件路径
    
    返回:
        list of dict: 每个检测结果包含 direction, distance, type, scene
    """
    global _model, CONF_THRESHOLD
    
    if _model is None:
        _init_model(bmodel_path)
    
    if conf_threshold is not None:
        CONF_THRESHOLD = conf_threshold
    
    input_tensor = _preprocess(image)
    
    if USE_SAIL:
        input_data = BMData(_handle, (1, 3, INPUT_SIZE, INPUT_SIZE))
        input_data.from_numpy(input_tensor.astype(np.uint8))
        outputs = _model.process([input_data])
        outputs = [out.to_numpy() for out in outputs]
    else:
        outputs = _model.run(None, {_model.get_inputs()[0].name: input_tensor})
    
    results = _postprocess(outputs, image.shape)
    return results


def get_obstacle_info(image, conf_threshold=0.5):
    """
    获取最近的障碍物信息（导盲场景主接口）
    
    返回格式:
        {
            "direction": "左前方",
            "distance": 1.2,
            "type": "person",
            "scene": "normal"
        }
        如果没有检测到障碍物，返回 None
    """
    detections = detect_objects(image, conf_threshold)
    
    if not detections:
        return None
    
    # 返回最近的障碍物（已按距离排序）
    return detections[0]


def get_all_obstacles(image, conf_threshold=0.5):
    """
    获取所有检测到的障碍物列表
    
    返回:
        list of dict，每个dict包含 direction, distance, type, scene
    """
    return detect_objects(image, conf_threshold)


def get_obstacles_by_type(image, target_type, conf_threshold=0.5):
    """
    获取指定类型的障碍物
    
    参数:
        target_type: 类别名称，如 'person', 'car'
    
    返回:
        list of dict
    """
    detections = detect_objects(image, conf_threshold)
    return [d for d in detections if d['type'] == target_type]


# ========== 辅助函数 ==========

def set_scene(scene):
    """
    设置场景类型
    scene: "normal", "night", "rain"
    """
    global SCENE_TYPE
    SCENE_TYPE = scene
    print(f"[INFO] 场景已切换为: {scene}")


def set_confidence_threshold(threshold):
    """
    设置置信度阈值
    """
    global CONF_THRESHOLD
    CONF_THRESHOLD = threshold
    print(f"[INFO] 置信度阈值已设置为: {threshold}")


def draw_detections(image, detections):
    """
    在图像上绘制检测结果（调试用）
    """
    for obj in detections:
        # 这里需要bbox信息，但当前输出没有bbox
        # 仅供调试，实际使用需要修改_postprocess保留bbox
        label = f"{obj['type']} {obj['distance']}m {obj['direction']}"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image


# ========== JSON输出函数 ==========

def get_obstacle_info_json(image, conf_threshold=0.5):
    """
    获取最近的障碍物信息（JSON字符串格式）
    方便C角直接使用
    """
    info = get_obstacle_info(image, conf_threshold)
    if info is None:
        return json.dumps({
            "direction": "无",
            "distance": None,
            "type": "none",
            "scene": SCENE_TYPE
        }, ensure_ascii=False)
    return json.dumps(info, ensure_ascii=False)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO Detector 测试")
    print("=" * 50)
    
    # 测试单张图片
    test_img_path = "test.jpg"
    import os
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        result = get_obstacle_info(img)
        print("\n[测试结果]")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试摄像头实时检测
    print("\n启动摄像头实时检测...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        exit(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 获取最近的障碍物信息
        obstacle = get_obstacle_info(frame)
        
        if obstacle:
            # 在图像上显示
            info_text = f"{obstacle['direction']} {obstacle['distance']}m {obstacle['type']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 控制台输出
            print(f"\r{info_text}", end="")
            
            # 判断是否需要报警（距离小于2米）
            if obstacle['distance'] < 2.0:
                cv2.putText(frame, "WARNING!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No obstacle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLO Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
