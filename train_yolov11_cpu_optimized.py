# -*- coding: utf-8 -*-
"""
YOLOv11 CPU优化训练脚本 - 红外小目标检测
针对CPU训练优化的参数配置

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
    
    # CPU优化训练参数
    model.train(
        data='configs/dataset_configs/tiaozhanbei_final.yaml',  # 数据集配置文件
        imgsz=416,                    # 降低图像尺寸以提高速度 (640->416)
        epochs=50,                    # 减少训练轮数进行快速验证
        batch=8,                      # 减小批处理大小以适应CPU内存
        workers=8,                    # 增加数据加载线程数
        device='cpu',                 # 明确指定使用CPU
        optimizer='AdamW',            # 使用AdamW优化器，通常收敛更快
        lr0=0.001,                    # 降低初始学习率
        lrf=0.01,                     # 最终学习率
        momentum=0.937,               # SGD动量
        weight_decay=0.0005,          # 权重衰减
        warmup_epochs=1,              # 减少热身轮数
        warmup_momentum=0.8,          # 热身动量
        warmup_bias_lr=0.1,           # 热身偏置学习率
        box=0.05,                     # box损失权重
        cls=0.5,                      # cls损失权重
        dfl=1.5,                      # dfl损失权重
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
        mosaic=0.5,                   # 减少马赛克增强以提高速度
        mixup=0.0,                    # 图像混合（概率）
        copy_paste=0.0,               # 分割复制粘贴（概率）
        close_mosaic=5,               # 在最后5个epoch关闭mosaic
        resume=False,                 # 从头开始训练
        project='runs/train',         # 项目保存目录
        name='yolov11_cpu_optimized', # 实验名称
        exist_ok=True,                # 覆盖现有项目/名称
        pretrained=True,              # 使用预训练模型
        verbose=True,                 # 详细输出
        seed=0,                       # 随机种子
        deterministic=True,           # 确定性训练
        single_cls=False,             # 训练单类检测器
        rect=False,                   # 矩形训练
        cos_lr=False,                 # 余弦学习率调度器
        fraction=0.5,                 # 使用50%的训练数据进行快速验证
        profile=False,                # 在训练期间分析ONNX和TensorRT速度
        freeze=None,                  # 冻结层：backbone=10, first3=0 1 2
        multi_scale=False,            # 多尺度训练，图像大小+/-50%
        overlap_mask=True,            # 训练期间分割掩码应重叠
        mask_ratio=4,                 # 掩码降采样比率
        dropout=0.0,                  # 使用dropout正则化
        val=True,                     # 训练期间验证/测试
        split='val',                  # 用于验证的数据集分割
        save_json=False,              # 将结果保存到JSON文件
        save_hybrid=False,            # 保存标签的混合版本
        conf=None,                    # 检测的对象置信度阈值
        iou=0.7,                      # NMS的交并比（IoU）阈值
        max_det=300,                  # 每张图像的最大检测数
        half=False,                   # 使用半精度（FP16）
        dnn=False,                    # 使用OpenCV DNN后端进行ONNX推理
        plots=True,                   # 训练期间保存绘图
        cache=True,                   # 缓存图像以提高训练速度
        save_period=10,               # 每10个epoch保存一次检查点
        patience=20,                  # 早停耐心值
    )
