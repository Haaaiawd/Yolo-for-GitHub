# -*- coding: utf-8 -*-
"""
YOLOv11 训练脚本 - 红外小目标检测
基于轻量化国产大模型的高帧频弱小目标检测识别技术研究

@Project: CQ-09 挑战杯
@Dataset: tiaozhanbei_datasets_final 
@Classes: 6 (drone, car, ship, bus, pedestrian, cyclist)
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化YOLOv11模型
    model = YOLO('models/yolo11n.pt')  # 使用预训练的YOLOv11 nano模型
    
    # 开始训练
    model.train(
        data='configs/dataset_configs/tiaozhanbei_final.yaml',  # 数据集配置文件
        imgsz=640,                    # 输入图像尺寸
        epochs=100,                   # 训练轮数
        batch=16,                     # 批处理大小（根据显存调整）
        workers=4,                    # 数据加载线程数
        device='',                    # 自动选择GPU/CPU
        optimizer='SGD',              # 优化器
        lr0=0.01,                     # 初始学习率
        lrf=0.01,                     # 最终学习率
        momentum=0.937,               # SGD动量
        weight_decay=0.0005,          # 权重衰减
        warmup_epochs=3,              # 热身轮数
        warmup_momentum=0.8,          # 热身动量
        warmup_bias_lr=0.1,           # 热身偏置学习率
        box=0.05,                     # box损失权重
        cls=0.5,                      # cls损失权重
        dfl=1.5,                      # dfl损失权重
        pose=12.0,                    # pose损失权重
        kobj=1.0,                     # keypoint obj损失权重
        label_smoothing=0.0,          # 标签平滑
        nbs=64,                       # 名义批处理大小
        hsv_h=0.015,                  # 图像HSV-Hue增强
        hsv_s=0.7,                    # 图像HSV-Saturation增强
        hsv_v=0.4,                    # 图像HSV-Value增强
        degrees=0.0,                  # 图像旋转（+/-度）
        translate=0.1,                # 图像平移（+/-分数）
        scale=0.5,                    # 图像缩放（+/-增益）
        shear=0.0,                    # 图像剪切（+/-度）
        perspective=0.0,              # 图像透视（+/-分数）
        flipud=0.0,                   # 图像上下翻转（概率）
        fliplr=0.5,                   # 图像左右翻转（概率）
        mosaic=1.0,                   # 图像马赛克（概率）
        mixup=0.0,                    # 图像混合（概率）
        copy_paste=0.0,               # 分割复制粘贴（概率）
        close_mosaic=10,              # 在最后10个epoch关闭mosaic
        resume=False,                 # 从头开始训练
        project='runs/train',         # 项目保存目录
        name='yolov11_infrared_detection',  # 实验名称
        exist_ok=True,               # 覆盖现有项目/名称
        pretrained=True,             # 使用预训练模型
        verbose=True,                # 详细输出
        seed=0,                      # 随机种子
        deterministic=True,          # 确定性训练
        single_cls=False,            # 训练单类检测器
        rect=False,                  # 矩形训练
        cos_lr=False,                # 余弦学习率调度器
        fraction=1.0,                # 训练数据集分数
        profile=False,               # 在训练期间分析ONNX和TensorRT速度
        freeze=None,                 # 冻结层：backbone=10, first3=0 1 2
        multi_scale=False,           # 多尺度训练，图像大小+/-50%
        overlap_mask=True,           # 训练期间分割掩码应重叠
        mask_ratio=4,                # 掩码降采样比率
        dropout=0.0,                 # 使用dropout正则化
        val=True,                    # 训练期间验证/测试
        split='val',                 # 用于验证的数据集分割
        save_json=False,             # 将结果保存到JSON文件
        save_hybrid=False,           # 保存标签的混合版本
        conf=None,                   # 检测的对象置信度阈值
        iou=0.7,                     # NMS的交并比（IoU）阈值
        max_det=300,                 # 每张图像的最大检测数
        half=False,                  # 使用半精度（FP16）
        dnn=False,                   # 使用OpenCV DNN后端进行ONNX推理
        plots=True,                  # 训练期间保存绘图
        source=None,                 # 推理的源目录
        vid_stride=1,                # 视频帧率步长
        stream_buffer=False,         # 缓冲所有流式帧
        visualize=False,             # 可视化特征
        augment=False,               # 应用图像增强到预测源
        agnostic_nms=False,          # 类别无关的NMS
        classes=None,                # 按类别过滤结果
        retina_masks=False,          # 使用高分辨率分割掩码
        embed=None,                  # 返回特征向量/嵌入
        show=False,                  # 显示预测的图像
        save_frames=False,           # 保存预测的单独帧
        save_txt=False,              # 将结果保存为.txt文件
        save_conf=False,             # 保存置信度到--save-txt标签
        save_crop=False,             # 保存裁剪的预测框
        show_labels=True,            # 显示预测标签
        show_conf=True,              # 显示预测置信度
        show_boxes=True,             # 显示预测框
        line_width=None,             # 边界框线宽度
    )
