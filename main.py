"""
单张图片检测版 main.py

功能：
1. 读取本地图片
2. 调用 A/B 黑匣子检测（yolo_detector）
3. 获取障碍物信息
4. 调用报警模块（alarm）
"""

import cv2
from alarm import AlarmCore
from speech import SpeechModule
from yolo_detector import get_all_obstacles


def main():
    print("===== 单张图片检测模式 =====")

    #初始化你的模块
    speech = SpeechModule()
    alarm = AlarmCore()

    alarm.start()
    speech.speak("开始图片检测")

    #读取图片
    image_path = "/Users/chenkeyao/Desktop/smarteye_guide/src/test.jpg"   #这里换成图片路径
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[错误] 图片读取失败：{image_path}")
        speech.speak("图片读取失败")
        return

    print("图片读取成功，开始检测")

    # 调用 A/B 检测黑匣子
    obstacle_list = get_all_obstacles(frame)

    # 输出检测结果
    if not obstacle_list:
        print("未检测到任何障碍物")
        speech.speak("未检测到障碍物")
    else:
        print(f"检测到 {len(obstacle_list)} 个目标")

        for obs in obstacle_list:
            print(obs)

            #调用你的报警系统
            alarm.trigger_obstacle(
                direction=obs.get("direction", "正前方"),
                distance=obs.get("distance", 999.0),
                obs_type=obs.get("type", "obstacle"),
                scene=obs.get("scene", "normal")
            )

    #结束
    alarm.stop()
    print("===== 检测结束 =====")


# 程序入口
if __name__ == "__main__":
    main()
