smarteye_guide/
│
├── README.md                    # 项目说明
├── requirements.txt             # pip install -r requirements.txt
│
├── detector_local.py            # ✅ 你写的，本地PyTorch版本
├── detector_remote.py           # ⏳ 以后A给你，调用少林派接口
├── detector.py                  # 🆕 统一入口，根据环境切换
│
├── mock_camera.py               # ✅ 你写的
├── real_camera.py               # 🆕 以后加，调用真实摄像头
│
├── mock_vibrator.py             # ✅ 你写的
├── real_vibrator.py             # 🆕 以后加，调用GPIO
│
├── voice.py                     # ✅ 你写的（需增加espeak）
│
├── main_local.py                # ✅ 你写的，本地测试用
├── test_integration.py          # ✅ 你写的，完整测试
│
├── images/                      # 测试图片文件夹
│   ├── bus.jpg
│   ├── person.jpg
│   └── ...
│
└── videos/                      # 🆕 测试视频文件夹
    └── test_walk.mp4
