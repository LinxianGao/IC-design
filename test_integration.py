import cv2
import time
from mock_camera import MockCamera
from mock_vibrator import MockVibrator
from voice import VoiceAlert
from detector_local import YOLODetector

class SmartEyeGuide:
    def __init__(self):
        print("="*50)
        print("  智瞳·导盲系统 - 真实YOLO检测版")
        print("="*50)
        print()
        
        # 初始化各模块
        self.camera = MockCamera(fps=3)  # 降低帧率，避免太快
        self.vibrator = MockVibrator()
        self.voice = VoiceAlert()
        self.detector = YOLODetector()  # ← 这里换成真实的YOLO检测器
        
        # 报警阈值（米）
        self.distance_threshold = 2.0
        
        # 统计
        self.frame_count = 0
        self.alert_count = 0
    
    def run(self, max_frames=20):
        """运行主循环"""
        print("🚀 系统启动\n")
        self.voice.speak("System started")
        
        while self.frame_count < max_frames:
            # 1. 获取图片
            ret, frame = self.camera.read()
            if not ret:
                print("获取图片失败")
                break
            
            self.frame_count += 1
            
            # 2. YOLO检测（这是关键！）
            detections = self.detector.detect(frame)
            
            # 3. 检查是否需要报警
            alert_triggered = False
            for det in detections:
                if det['distance'] < self.distance_threshold:
                    alert_triggered = True
                    print(f"\n⚠️ 检测到 {det['label']} 在 {det['distance']}米处 (置信度:{det['confidence']})")
                    
                    # 语音播报
                    self.voice.speak(f"Warning, {det['label']} ahead")
                    
                    # 震动反馈（距离越近震动越长）
                    if det['distance'] < 1.0:
                        self.vibrator.vibrate(1.0)
                    elif det['distance'] < 1.5:
                        self.vibrator.vibrate(0.5)
                    else:
                        self.vibrator.vibrate(0.3)
                    
                    self.alert_count += 1
                    break  # 一次只报警第一个障碍物
            
            # 4. 打印状态
            if not alert_triggered:
                if self.frame_count % 5 == 0:
                    print(f"📸 第{self.frame_count}帧: 安全 (检测到{len(detections)}个物体)")
            else:
                print(f"📸 第{self.frame_count}帧: ⚠️ 已报警")
            
            # 控制帧率
            time.sleep(0.1)
        
        # 结束
        self.camera.release()
        print(f"\n📊 统计: 共处理 {self.frame_count} 帧, 触发报警 {self.alert_count} 次")
        self.voice.speak("System stopped")


# 可选：带画面显示的版本
class SmartEyeGuideWithDisplay(SmartEyeGuide):
    def run(self, max_frames=50):
        """带实时画面显示的版本"""
        print("🚀 系统启动（带画面显示）\n")
        self.voice.speak("System started")
        
        while self.frame_count < max_frames:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # 检测并画框
            display_frame, detections = self.detector.detect_and_draw(frame)
            
            # 检查报警
            alert_triggered = False
            for det in detections:
                if det['distance'] < self.distance_threshold:
                    alert_triggered = True
                    print(f"⚠️ {det['label']} {det['distance']}米")
                    self.voice.speak(f"Warning, {det['label']} ahead")
                    self.vibrator.vibrate(0.5)
                    self.alert_count += 1
                    break
            
            # 显示画面
            cv2.imshow('SmartEye Guide', display_frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.05)
        
        self.camera.release()
        cv2.destroyAllWindows()
        print(f"\n统计: {self.frame_count}帧, {self.alert_count}次报警")
        self.voice.speak("System stopped")


if __name__ == "__main__":
    # 运行标准版（无画面）
    guide = SmartEyeGuide()
    guide.run(max_frames=15)
    
    # 如果想看实时画面，用下面这个：
    # guide = SmartEyeGuideWithDisplay()
    # guide.run(max_frames=50)