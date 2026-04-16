"""
震动执行模块：
仅负责根据指令震动，不参与任何逻辑决策
适配：实际嵌入式GPIO
"""
import time

class VibrateModule:
    def __init__(self):
        self.module_name = "震动马达"

    # 基础震动：强度1~4，时长秒
    def vibrate(self, strength: int, duration: float):
        print(f"[震动] 强度:{strength} 时长:{duration}s 开始")
        # 真实硬件在这里控制GPIO
        time.sleep(0.1)  # 模拟执行
        print(f"[震动] 结束")

    # 短脉冲震动（提醒用）
    def pulse_short(self):
        print(f"[震动] 短脉冲 ⚪")
        time.sleep(0.1)

    # 长脉冲震动（警告用）
    def pulse_long(self):
        print(f"[震动] 长脉冲 🔴")

    # 连续急促震动（特级危险）
    def pulse_emergency(self):
        print(f"[震动] 急促连续 ⛔")
