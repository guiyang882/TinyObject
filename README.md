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
.
├── README.md
├── mainmodels
│   ├── dataset
│   └── models
└── source
    ├── bg
    ├── test
    ├── tools
    └── tracking
```

### 模型目录
```bash
.
├── dataset
│   ├── NWPU_dataset_input.py
│   ├── TT100k_traffic_sign_input.py
│   ├── airplane_input.py
│   ├── cifar_input.py
│   ├── show_target.py
│   ├── tools.py
│   └── usa_traffic_sign_input.py
└── models
    ├── autoencoder
    ├── resnet
    ├── rpnplus
    ├── ssd
    └── tradition
```

---

## TODOLIST
- [x] 对JS1号卫星图像数据进行裁剪，并将相应的目标的位置进行坐标变换
- [x] 使用C++编写SSD网络中不同feature_map中对应的anchor的标签数据，这其中问题较多
- [x] 在构造anchor boxes时，需要将inner的box剔除出去
- [x] 对于随机采样出来的图像，在生成SSD训练样本时，却发现仅仅由大约 30% 的图像能在不同的feature map中找到对应的候选框
- [x] 将物体覆盖的率的评价函数由IOU改成overlap-ratio
- [x] 使用Python将anchor在没有获得候选框的图像的结果可视化出来，分析在不同目标尺度上的差异
- [x] 对于上述结果进行可视化分析，找到适合的阈值，使得SSD模型能覆盖大部分目标
- [ ] 现在训练SSD网络，base网络是自己定义的普通的CNN网络
- [ ] 紧接着将ResNet作为base网络，对比模型结果
