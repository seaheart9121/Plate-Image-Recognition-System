项目简介
本项目基于Python+PyTorch+OpenCV开发智能停车场车牌识别系统，依托 YOLOv8-Pose 关键点检测、透视变换、LPRNet+CTC 时序识别、ColorNet 颜色分类四大技术，实现实景车辆车牌畸变矫正、字符识别、车牌颜色判定，配套模拟停车场入场登记、车位分配、自动计费全业务逻辑，可本地离线部署，适配蓝牌、黄牌、新能源绿牌多车型，兼容逆光、倾斜、轻微遮挡等复杂实景环境。

在终端中执行以下命令
```bash
pip install -r requirements.txt
```

运行main.py

四大核心模块说明
车牌关键点检测模块（YOLOv8-Pose）
读取实景图片，检测车牌四个角点坐标，依靠单应性矩阵做透视变换，矫正倾斜、透视畸变车牌，解决实拍变形无法识别痛点；训练输出 loss/P/R/mAP/PR 曲线，通过混淆矩阵评估关键点检测精度。
车牌字符识别模块（LPRNet+CTC）
轻量化卷积网络搭配 CTC 损失，无需手动切割字符，端到端识别 7 位普通蓝牌、8 位新能源绿牌不定长字符。
车牌颜色分类模块（ColorNet）
3 层轻量化 CNN 网络，自动区分蓝、黄、绿三类车牌；通过convertScaleAbs(alpha,beta)调整亮度对比度做样本扩充，改善黄牌样本稀缺问题。
停车场业务模块
集成车辆进场登记、车位统计、自动计费、出场结算，全流程闭环管理，系统本地离线运行，无云端数据上传，保障车牌隐私数据安全。

数据准备：ccpd_to_yolo_pose.py → 
模型训练：train_pose.py → 
图片采集：opencvutil.py → 
车牌矫正：predict_pose.py → 
核心识别：plate_recognizer.py（+ ocrutil.py兜底）→ 
数据管理：datautil.py → 
可视化交互：main.py

CCPD → 数据转换 → LPRNet训练 → exp对比 → OCR识别
YOLO检测 → 车牌裁剪 → LPRNet识别 → 输出结果
车牌图片 → CNN分类 → 蓝/黄/绿