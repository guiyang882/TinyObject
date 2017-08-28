# TinyObject
主要是用来解决小目标的检测和跟踪问题

## 数据来源
- Skybox Imaging Satellite Video Dataset 
- 斯坦福无人机航拍数据
- 吉林一号视频数据

---

## 工程的主要目录结构
### 工程根目录
```bash
├── LearnerDetector
├── README.md
├── attached
├── output
├── source
└── trainmodel
```

---

## TODOLIST
[] 对JS1号卫星图像数据进行裁剪，并将相应的目标的位置进行坐标变换
[] 使用C++编写SSD网络中不同feature_map中对应的anchor的标签数据，这其中问题较多
[] 使用Python将anchor在没有获得候选框的图像的结果可视化出来，分析在不同目标尺度上的差异