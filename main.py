# 智瞳·导盲系统 主控程序 - 项目核心入口
"""
【标准障碍物数据格式】
    obstacle = {
        "direction": "左前方",
        "distance": 1.2,
        "type": "person",
        "scene": "normal"
    }
"""

import time
from speech import SpeechModule
from alarm import AlarmCore

# 正式联调时，在这里接入 A / B 的真实模块
# from camera import CameraModule         # B开发：摄像头采集模块
# from inference import InferenceModule   # A开发：算法推理模块


class SmartEyeGuide:
    """
    系统主控类：
    作为整个导盲系统的“总指挥中心”，统一调度各功能模块
    """

    def __init__(self):
        print("===== 智瞳·导盲系统 启动中 =====")

        # 初始化核心模块
        self.speech = SpeechModule()
        self.alarm = AlarmCore()

        # 预留模块占位（A/B接入前默认为空）
        self.camera = None
        self.inference = None

        # 系统运行状态
        self.running = False

        # 主循环帧率控制
        self.FRAME_INTERVAL = 0.1

        # 系统初始化
        self.system_init()

    def system_init(self):
        """
        系统初始化：
        仅负责系统启动、模块就绪提示
        """
        try:
            self.alarm.start()
            print("系统初始化完成，等待启动运行")
        except Exception as e:
            print(f"[初始化异常] {e}")

    # =========================
    # A / B 模块绑定接口
    # =========================
    def bind_camera_module(self, camera_module):
        """
        绑定 B 开发的摄像头模块
        要求：至少提供 get_frame() 方法
        """
        self.camera = camera_module
        print("摄像头模块绑定成功")

    def bind_inference_module(self, inference_module):
        """
        绑定 A 开发的算法推理模块
        推荐接口：
            detect_obstacle(frame) -> dict 或 list[dict]
        """
        self.inference = inference_module
        print("算法推理模块绑定成功")

    # 主运行入口
    def run(self):
        """
        系统运行总入口
        逻辑：
            1. 未绑定 A/B 模块 -> 进入模拟模式
            2. 已绑定 A/B 模块 -> 进入正式模式
        """
        if not self.camera or not self.inference:
            print("警告：摄像头或算法模块未绑定，进入模拟运行模式")
            self.speech.speak("进入模拟运行模式")
            self.running = True
            self._mock_run_loop()
            return

        print("A/B模块已绑定，进入正式运行模式")
        self.speech.speak("系统开始运行")
        self.running = True
        self._formal_run_loop()

    def stop_system(self):
        """
        停止系统运行
        """
        self.running = False
        try:
            self.alarm.stop()
        except Exception as e:
            print(f"[停止异常] {e}")
        print("===== 智瞳·导盲系统 已停止 =====")

    # 正式运行模式
    def _formal_run_loop(self):
        """
        正式运行主循环：
            摄像头采集 -> 算法推理 -> 结果标准化 -> 调用报警模块
        """
        while self.running:
            try:
                # Step 1：B模块采集一帧画面
                frame = self.camera.get_frame()

                # Step 2：A模块进行障碍物检测/推理
                # 这里是A的核心拼接位置
                raw_result = self.inference.detect_obstacle(frame)

                # Step 3：统一标准化 A 输出的数据格式
                obstacle_list = self._normalize_inference_result(raw_result)

                # Step 4：逐个障碍物触发报警决策
                for obstacle in obstacle_list:
                    self.alarm.trigger_obstacle(
                        direction=obstacle["direction"],
                        distance=obstacle["distance"],
                        obs_type=obstacle["type"],
                        scene=obstacle["scene"]
                    )

                # Step 5：控制循环频率，防止系统资源占用过高
                time.sleep(self.FRAME_INTERVAL)

            except KeyboardInterrupt:
                print("\n检测到手动终止，系统准备关闭")
                self.stop_system()

            except Exception as e:
                print(f"[运行异常] {e}")
                # 单帧异常兜底：不中断整个系统
                time.sleep(0.5)


    # 数据标准化
    def _normalize_inference_result(self, raw_result):
        """
        将 A 模块输出统一转换为标准障碍物列表格式

        兼容情况：
            1. A 返回单个字典 dict
            2. A 返回字典列表 list[dict]
            3. A 返回字段不完整 -> 自动补默认值
            4. A 返回大小写/别名不统一 -> 自动映射标准化

        标准输出格式：
            [
                {
                    "direction": "左前方",
                    "distance": 1.2,
                    "type": "person",
                    "scene": "normal"
                }
            ]
        """
        if raw_result is None:
            return []

        # 若 A 返回的是单个障碍物字典，自动包装成列表
        if isinstance(raw_result, dict):
            raw_result = [raw_result]

        # 非法格式兜底
        if not isinstance(raw_result, list):
            print("[数据警告] 推理结果不是 dict/list，已忽略")
            return []

        normalized_list = []

        for item in raw_result:
            if not isinstance(item, dict):
                continue

            # 提取并标准化字段
            direction = self._normalize_direction(item.get("direction"))
            distance = self._normalize_distance(item.get("distance"))
            obs_type = self._normalize_type(item.get("type"))
            scene = self._normalize_scene(item.get("scene"))

            normalized_list.append({
                "direction": direction,
                "distance": distance,
                "type": obs_type,
                "scene": scene
            })

        return normalized_list

    def _normalize_distance(self, raw_distance):
        """
        标准化距离数据：
            - 转为 float
            - 非法值统一兜底为安全远距离
        """
        try:
            distance = float(raw_distance)
            if distance < 0:
                return 999.0
            return distance
        except (TypeError, ValueError):
            return 999.0

    def _normalize_type(self, raw_type):
        """
        标准化障碍物类别：
            - 统一为小写英文标签
            - 处理大小写、别名、异常值
            - 最终交给 speech 模块映射中文播报
        """
        if not isinstance(raw_type, str):
            return "obstacle"

        t = raw_type.strip().lower()

        # A可能输出的不同类别命名，统一收口
        if t in ["person", "people", "human", "pedestrian"]:
            return "person"
        elif t in ["car", "vehicle", "bus", "truck"]:
            return "car"
        elif t in ["bicycle", "bike", "cycle", "motorcycle"]:
            return "bicycle"
        else:
            return "obstacle"

    def _normalize_direction(self, raw_direction):
        """
        标准化方向字段：
            内部统一使用中文方向短语，便于直接播报
        """
        if not isinstance(raw_direction, str):
            return "正前方"

        d = raw_direction.strip().lower()

        direction_map = {
            "left": "左侧",
            "left_front": "左前方",
            "front_left": "左前方",
            "front": "正前方",
            "center": "正前方",
            "right": "右侧",
            "right_front": "右前方",
            "front_right": "右前方",

            "左": "左侧",
            "左侧": "左侧",
            "左前": "左前方",
            "左前方": "左前方",

            "前": "正前方",
            "正前": "正前方",
            "正前方": "正前方",

            "右": "右侧",
            "右侧": "右侧",
            "右前": "右前方",
            "右前方": "右前方",
        }

        return direction_map.get(d, "正前方")

    def _normalize_scene(self, raw_scene):
        """
        标准化场景字段：
            scene 最终只保留以下三类：
                1. normal
                2. blind_road
                3. road_abnormal
        """
        if not isinstance(raw_scene, str):
            return "normal"

        s = raw_scene.strip().lower()

        if s in ["blind_road", "blindroad", "blind-road", "盲道"]:
            return "blind_road"
        elif s in ["road_abnormal", "roadabnormal", "road-abnormal", "abnormal_road", "路面异常"]:
            return "road_abnormal"
        else:
            return "normal"


    # 模拟运行模式
    def _mock_run_loop(self):
        """
        模拟运行循环：
        无需 A/B 模块、无需硬件，也能验证完整报警链路
        """
        test_scenes = [
            {"direction": "左前方", "distance": 2.5, "type": "person", "scene": "normal"},
            {"direction": "正前方", "distance": 1.2, "type": "car", "scene": "normal"},
            {"direction": "右侧", "distance": 0.6, "type": "bicycle", "scene": "normal"},
            {"direction": "正前方", "distance": 1.0, "type": "obstacle", "scene": "blind_road"},
            {"direction": "左侧", "distance": 1.5, "type": "obstacle", "scene": "road_abnormal"},
        ]

        for obstacle in test_scenes:
            if not self.running:
                break

            print(f"\n[模拟检测] {obstacle}")

            try:
                self.alarm.trigger_obstacle(
                    direction=obstacle["direction"],
                    distance=obstacle["distance"],
                    obs_type=obstacle["type"],
                    scene=obstacle["scene"]
                )
            except Exception as e:
                print(f"[模拟运行异常] {e}")

            time.sleep(3)

        self.stop_system()


# =========================
# 以下为联调前测试用的模拟模块
# A/B 真模块接入后可直接删除
# =========================
class MockCameraModule:
    """
    模拟摄像头模块：
    用于在无真实硬件时测试主控流程
    """
    def get_frame(self):
        return "mock_frame"


class MockInferenceModule:
    """
    模拟算法推理模块：
    模拟 A 返回障碍物检测结果
    """
    def __init__(self):
        self.index = 0
        self.test_data = [
            {"direction": "left_front", "distance": 2.2, "type": "person", "scene": "normal"},
            {"direction": "front", "distance": 0.7, "type": "car", "scene": "normal"},
            {"direction": "right", "distance": 1.1, "type": "bike", "scene": "normal"},
            {"direction": "front", "distance": 1.0, "type": "obstacle", "scene": "blind_road"},
            {"direction": "left", "distance": 1.4, "type": "obstacle", "scene": "road_abnormal"},
        ]

    def detect_obstacle(self, frame):
        """
        模拟 A 的接口：
            detect_obstacle(frame) -> list[dict]
        """
        result = self.test_data[self.index]
        self.index = (self.index + 1) % len(self.test_data)
        return [result]



# 程序入口
if __name__ == "__main__":
    guide_system = SmartEyeGuide()

    # 方案1：直接运行（未绑定A/B时自动进入模拟模式）
    guide_system.run()


    # 方案2：联调时使用（把上面那行 guide_system.run() 注释掉）

    # guide_system = SmartEyeGuide()
    #
    # ===== 拼接 B 的代码位置 =====
    # camera_module = CameraModule()
    # guide_system.bind_camera_module(camera_module)
    #
    # ===== 拼接 A 的代码位置 =====
    # inference_module = InferenceModule()
    # guide_system.bind_inference_module(inference_module)
    #
    # ===== 正式启动系统 =====
    # guide_system.run()


    # 方案3：本地伪联调测试（可选）
    # guide_system = SmartEyeGuide()
    # guide_system.bind_camera_module(MockCameraModule())
    # guide_system.bind_inference_module(MockInferenceModule())
    # guide_system.run()
