# GreedyTrack

**计算机视觉基础课程大作业**

组员：**陈卓，李强**

指导老师：**姜志宇**

## 整体结构

![](images\框架.png)

## 追踪流程：

<img src="images\算法.png" style="zoom:60%;" />

## 使用

```c++
git clone git@github.com:cpeterz/GreedyTrack.git
cd GreedyTrack
python train.py   // 训练
python detect.py  // 检测
python track.py   // 追踪
```

环境需求：torch torchvision等常用深度学习库

数据集：Jiang N, Wang K, Peng X, et al. Anti-uav: a large-scale benchmark for vision-based uav tracking[J]. IEEE Transactions on Multimedia, 2021, 25: 486-500.

