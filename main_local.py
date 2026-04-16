import cv2
import time
from mock_camera import MockCamera
from mock_vibrator import MockVibrator
from voice_local import VoiceAlert
from detector_local import YOLODetector

class SmartEyeGuide:
    def __init__(self):
        print("="*50)
        print("智瞳·导盲系统 - 本地完整版")
        print("="*50)
        
        self.camera = MockCamera()
        self.vibrator = MockVibrator()
        self.voice = VoiceAlert()
        self.detector = YOLODetector()
        self.threshold = 2.0
        
    def run(self):
        self.voice.speak("System started")
        frame_count = 0
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 调用YOLO检测
            detections = self.detector.detect(frame)
            
            # 检查是否有障碍物在阈值内
            alert_triggered = False
            for det in detections:
                if det['distance'] < self.threshold:
                    print(f"⚠️ {det['label']} 在 {det['distance']:.1f}米处")
                    self.voice.speak(f"Warning, {det['label']} ahead")
                    self.vibrator.vibrate(0.5)
                    alert_triggered = True
                    break
            
            if not alert_triggered and frame_count % 10 == 0:
                print(f"第{frame_count}帧: 安全")
            
            # 按q退出（如果有显示窗口）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        self.voice.speak("System stopped")

if __name__ == "__main__":
    guide = SmartEyeGuide()
    guide.run()