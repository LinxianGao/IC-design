import sys
import json
import cv2

sys.path.insert(0, '/data/pythoncode')

try:
    from yolo_detector import YOLODetector
except ImportError:
    print(json.dumps({"error": "无法导入 yolo_detector"}))
    sys.exit(1)

# 配置
bmodel_path = "/data/dataset/bmodels/ten_agu_f32.bmodel"
test_image = "/data/dataset/runs/detect/盲道.webp"

# 初始化
detector = YOLODetector(bmodel_path, conf_thresh=0.5)

# 读取图片
img = cv2.imread(test_image)
if img is None:
    print(json.dumps({"error": f"无法读取图片: {test_image}"}))
    sys.exit(1)

# 感知
perception_result, _ = detector.perceive(img, with_ocr=False)

# 只输出字典
print(json.dumps(perception_result, ensure_ascii=False))
