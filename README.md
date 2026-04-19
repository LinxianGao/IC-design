smarteye_guide/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包列表
│
├── src/                         # 源代码目录
│   ├── main.py                  # 【C负责】主程序入口
│   ├── camera.py                # 【B负责】摄像头采集
│   ├── vibrate.py               # 【B负责】震动马达控制
│   ├── speech.py                # 【C负责】语音播报
│   │
│   ├── models/                  # 模型目录
│   │   ├── yolov8n_int8.bmodel      # 【A负责】目标检测模型
│   │   ├── pidnet_int8.bmodel       # 【A负责】盲道分割模型
│   │   └── depth_int8.bmodel        # 【A负责】深度估计模型
│   │
│   ├── detectors/               # 推理模块目录
│   │   ├── yolo_detector.py         # 【A负责】YOLO推理封装
│   │   ├── segmentor.py             # 【A负责】分割推理封装
│   │   └── depth_estimator.py       # 【A负责】深度推理封装
│   │
│   └── utils/                   # 工具模块
│       ├── geometry.py              # 【C负责】几何计算（重叠判断、距离计算）
│       └── logger.py                # 【C负责】日志记录
│
├── tests/                       # 测试脚本
│   ├── test_camera.py           # 【B负责】
│   ├── test_vibrate.py          # 【B负责】
│   └── test_integration.py      # 【C负责】
│
└── config/                      # 配置文件
    └── settings.py              # 【C负责】所有阈值参数
