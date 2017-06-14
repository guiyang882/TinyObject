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

### 局部道路标记文件marklane.py
```bash
key w: save the line_th
//key n: start the new line
//key c: clear the cur line data, which have labled
key q: close the figure window
```

- 绘制道路曲线示意图

<img src="https://github.com/liuguiyangnwpu/TinyObject/blob/master/attached/drawLine.png" width="300" height="300"/>

**PS:**
所有的图像处理以及道路的标注都是按照左上角为坐标远点进行计算

---

### [引入dytb网络模型库](https://github.com/galeone/dynamic-training-bench)
- 使用的网络模型结构如下所示：

<img src="https://github.com/liuguiyangnwpu/TinyObject/blob/master/attached/SAEModel.png" width="600" height="400"/>

- 使用dytb模型库进行修改并加以训练，得到基于CIFAR10数据的SingleCAE网络，其自动编码机的结果如下：

<img src="https://github.com/liuguiyangnwpu/TinyObject/blob/master/attached/CIFAR10train.png" width="600" height="300"/>

- 经过对于SAE模型的微调，使用自己的数据得到如下结果：

<img src="https://github.com/liuguiyangnwpu/TinyObject/blob/master/attached/airplane.png" width="500" height="300"/>

- [Convolutional Autoencoders](https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/)
---

### [引入OpenTLD库进行专家学习](https://github.com/liuguiyangnwpu/LearnerDetector)
具体的使用规则见对应工程的Readme