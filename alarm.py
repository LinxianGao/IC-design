"""
智瞳导盲系统 - 报警决策核心
【核心决策能力】
    1. 输入：原始障碍物信息（方向+距离+类型+场景）
    2. 决策：自动计算危险等级、判断是否报警、判断报警优先级
    3. 输出：调用语音模块执行最终播报
"""
import time
from speech import SpeechModule
from mock_vibrate import VibrateModule

class AlarmCore:
    """
    报警决策中枢：具备完整逻辑判断、优先级控制、防抖限流
    所有业务规则全部写在这里，是系统的核心大脑
    """
    def __init__(self):
        #初始化语音输出模块
        self.speech = SpeechModule()
        self.vibrate = VibrateModule();

        #决策规则：距离阈值
        self.LEVEL4_DANGER = 0.5    # 特级危险 <0.5m
        self.LEVEL3_EMERGENCY = 0.8 # 紧急危险 0.5~0.8m
        self.LEVEL2_WARNING = 1.5   # 注意危险 0.8~1.5m
        self.LEVEL1_REMIND = 3.0    # 常规提醒 1.5~3.0m

        #决策规则：防抖
        self.last_alert_time = 0
        self.ALERT_INTERVAL = 2     # 2秒内不重复报警

    #内部决策方法
    def __is_allowed_to_alert(self) -> bool:
        """决策1：是否允许报警（防抖判断）"""
        now = time.time()
        if now - self.last_alert_time >= self.ALERT_INTERVAL:
            self.last_alert_time = now
            return True
        return False

    def __calculate_danger_level(self, distance: float) -> int:
        """决策2：根据距离自动计算危险等级"""
        if not isinstance(distance, (int, float)) or distance < 0:
            return 0  #决策：非法数据→安全等级
        if distance < self.LEVEL4_DANGER:
            return 4
        elif distance < self.LEVEL3_EMERGENCY:
            return 3
        elif distance < self.LEVEL2_WARNING:
            return 2
        elif distance < self.LEVEL1_REMIND:
            return 1
        else:
            return 0

    def __check_scene_priority(self, scene: str) -> bool:
        """决策3：特殊场景是否优先播报"""
        return scene in ["blind_road", "road_abnormal"]

    def start(self):
        """系统启动（无决策，仅初始化）"""
        self.speech.system_start()

    def stop(self):
        """系统停止（无决策，仅关闭）"""
        self.speech.system_stop()
        
    #震动模式决策
    def __decision_vibrate(self, level: int):
        """
        真正决策：根据危险等级 自动选震动模式
        level 0~4
        """
        if level == 4:  # 特级危险：连续急促
            self.vibrate.pulse_emergency()
        elif level == 3:  # 紧急危险：长脉冲
            self.vibrate.pulse_long()
        elif level == 2:  # 警告：短脉冲
            self.vibrate.pulse_short()
        elif level == 1:  # 提醒：轻微震动
            self.vibrate.vibrate(1, 0.2)
        else:
            pass  # 安全：不震动

    def trigger_obstacle(self, direction: str, distance: float, obs_type: str, scene: str = "normal"):
        """
        【唯一对外入口】
        接收算法原始数据→内部完成所有决策→输出语音
        :param direction:方向
        :param distance:距离（原始数据）
        :param obs_type:类型
        :param scene:场景
        """
        # 决策1：防抖 → 不满足则直接返回
        if not self.__is_allowed_to_alert():
            return

        # 决策2：计算危险等级
        level = self.__calculate_danger_level(distance)

        # 决策3：场景优先级判断
        if self.__check_scene_priority(scene):
            if scene == "blind_road":
                self.speech.warn_blind_road(direction)
            elif scene == "road_abnormal":
                self.speech.warn_road_abnormal(direction, distance)
            return

        # 最终决策：执行障碍物播报
        self.speech.warn_obstacle_full(direction, distance, obs_type, level)
        self.__decision_vibrate(level)


#测试
if __name__ == "__main__":
    alarm = AlarmCore()
    alarm.start()
    alarm.trigger_obstacle("左前方", 0.4, "person")
    alarm.trigger_obstacle("正前方", 0.9, "car")
    alarm.trigger_obstacle("右侧", 2.2, "obstacle")
    alarm.trigger_obstacle("正前方", 1.0, "obstacle", scene="blind_road")
    alarm.trigger_obstacle("左侧", 1.5, "obstacle", scene="road_abnormal")
    alarm.stop()
