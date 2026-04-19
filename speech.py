"""
智瞳导盲系统 - 语音播报核心模块（工程整合版）
【核心设计】
    打破模块拆分，将「方向+距离+障碍物类型+危险等级」
【规范遵循】
    团队V1.0标准：4类障碍物、5级距离阈值、3方向预警、2类特殊场景
【场景优化】
    盲道：主动指引沿盲道行走（真实导盲逻辑）
    路面异常：提示位置+距离+绕行建议
    常规障碍：方向+距离+类型+危险等级+行动指令，一体化输出
【适配平台】macOS/BM1684 Linux开发板
"""
import sys
import os


class SpeechModule:
    """
    一体化语音播报类
    核心能力：结合方向+距离+类型+等级+指引，一次性完整输出
    """
    def __init__(self):
        """
        初始化语音模块：自动识别平台 + 配置最优播报参数
        """
        # 平台自动识别（mac/开发板）
        self.is_mac = sys.platform == "darwin"
        self.is_linux = sys.platform == "linux"

        # 导盲专用语音参数（人声自然、清晰、不急促）
        self.mac_speed = 170
        self.linux_speed = 115
        self.linux_volume = 100

    def speak(self, text: str) -> None:
        """
        核心播报入口：统一日志 + 双平台适配 + 异常安全兜底
        :param text: 完整播报语句（已整合所有信息）
        """
        print(f"[语音播报] {text}")
        try:
            if self.is_mac:
                os.system(f'say -r {self.mac_speed} "{text}"')
            elif self.is_linux:
                os.system(f'ekho -s {self.linux_speed} -a {self.linux_volume} "{text}"')
        except Exception:
            print("[语音异常] 播报失败，系统继续运行")

    # 核心整合方法
    def warn_obstacle_full(self, direction: str, distance: float, obs_type: str, level: int) -> None:
        """
        【一体化核心播报】方向+距离+类型+危险等级+行动指引
        :param direction: 方位（左前方/右前方/正前方）
        :param distance: 距离（米）
        :param obs_type: 类型（person/car/bicycle/obstacle）
        :param level: 危险等级 0-4
        """
        # 类型映射（转为自然语言）
        type_map = {
            "person":"行人",
            "car": "车辆",
            "bicycle": "非机动车",
            "obstacle": "障碍物"
        }
        obs_name = type_map.get(obs_type, "障碍物")
        dist_str = f"{distance:.1f}"

        # 按等级输出
        if level == 4:
            self.speak(f"{direction} {dist_str}米{obs_name}，特级危险！请立即停下！")
        elif level == 3:
            self.speak(f"{direction} {dist_str}米{obs_name}，紧急危险！请立刻停下！")
        elif level == 2:
            self.speak(f"{direction} {dist_str}米{obs_name}，注意危险！请小心通行！")
        elif level == 1:
            self.speak(f"{direction} {dist_str}米{obs_name}，请注意避让！")
        else:
            self.speak(f"前方安全，可正常行走")

    #特殊场景
    def warn_blind_road(self, direction: str = "正前方") -> None:
        """
        盲道识别：真实导盲指引不是空泛提示）
        逻辑：识别到盲道 → 主动引导沿盲道行走
        """
        self.speak(f"{direction}为盲道，请沿盲道直行，保障行走安全")

    def warn_road_abnormal(self, direction: str, distance: float) -> None:
        """
        路面异常：**位置+距离+绕行指引**（实用化）
        """
        self.speak(f"{direction} {distance:.1f}米路面异常，请绕行通过")

    #系统状态
    def system_start(self) -> None:
        """系统启动：友好欢迎语"""
        self.speak("智瞳导盲系统已启动，正在为您保驾护航")

    def system_stop(self) -> None:
        """系统关闭：安全提示"""
        self.speak("导盲系统已关闭，请注意周边环境")
if __name__ == "__main__":
    test=SpeechModule()
    test.system_start()
    test.warn_obstacle_full("左前方",0.9,"person",2)
    test.warn_blind_road()
    test.warn_blind_road("右前方")
    test.warn_road_abnormal("左前方",1.0)
    test.system_stop()
