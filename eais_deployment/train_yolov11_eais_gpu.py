# -*- coding: utf-8 -*-
"""
YOLOv11 阿里云EAIS GPU训练脚本 - 红外小目标检测
针对魔搭社区阿里云弹性加速计算EAIS实例优化

@Project: CQ-09 挑战杯
@Dataset: tiaozhanbei_datasets_final 
@Classes: 6 (drone, car, ship, bus, pedestrian, cyclist)
@Hardware: 阿里云EAIS GPU实例
"""
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import sys
from ultralytics import YOLO

def check_gpu_environment():
    """检查GPU环境配置"""
    print("🔍 检查GPU环境...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        return True
    else:
        print("❌ 未检测到GPU，将使用CPU训练")
        return False

def optimize_gpu_settings():
    """优化GPU设置"""
    if torch.cuda.is_available():
        # 启用GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        print("✅ GPU优化设置已启用")

def setup_training_environment():
    """设置训练环境"""
    # 创建必要的目录
    os.makedirs('runs/train', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    # 设置环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步GPU操作
    os.environ['OMP_NUM_THREADS'] = '8'       # CPU线程数
    
    print("📁 训练环境已设置")

if __name__ == '__main__':
    print("🚀 开始YOLOv11 阿里云EAIS GPU训练")
    print("=" * 60)
    
    # 1. 检查GPU环境
    gpu_available = check_gpu_environment()
    
    # 2. 优化GPU设置
    optimize_gpu_settings()
    
    # 3. 设置训练环境
    setup_training_environment()
    
    # 4. 初始化YOLOv11模型
    print("\n📥 加载YOLOv11x预训练模型...")
    model = YOLO('yolo11x.pt')  # 切换到x版本
    
    # 5. 配置训练参数 - 针对阿里云EAIS GPU优化
    print("\n📝 配置训练参数...")
    
    # 根据GPU可用性调整设备和批次大小
    if gpu_available:
        device = 'cuda'
        batch_size = 8  # 修正: x版本模型较大，必须减小批次大小以防OOM
        workers = 8      # 更多的数据加载线程
        amp = True       # 启用混合精度训练
        print("🚀 使用GPU训练配置 (YOLOv11x - 小批次)")
    else:
        device = 'cpu'
        batch_size = 4  # CPU模式下也相应减小
        workers = 4
        amp = False
        print("⚠️ 使用CPU训练配置")
    
    training_args = {
        # 数据配置
        # 使用按序列划分的、更科学的数据集配置文件
        'data': 'configs/dataset_configs/tiaozhanbei_sequenced_x.yaml', 
        'imgsz': 640,                      # 标准图像尺寸
        'epochs': 100,                     # 更多训练轮次
        'batch': batch_size,
        'workers': workers,
        
        # 设备配置
        'device': device,
        'amp': amp,                        # 自动混合精度
        
        # 优化器配置
        'optimizer': 'AdamW',
        'lr0': 0.01,                       # 初始学习率
        'lrf': 0.01,                       # 最终学习率因子
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,                # 热身轮数
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 损失函数权重
        'box': 7.5,                        # box损失权重
        'cls': 0.5,                        # 分类损失权重
        'dfl': 1.5,                        # DFL损失权重
        
        # 数据增强
        'hsv_h': 0.015,                    # HSV色调增强
        'hsv_s': 0.7,                      # HSV饱和度增强
        'hsv_v': 0.4,                      # HSV明度增强
        'degrees': 0.0,                    # 旋转增强
        'translate': 0.1,                  # 平移增强
        'scale': 0.5,                      # 缩放增强
        'shear': 0.0,                      # 剪切增强
        'perspective': 0.0,                # 透视增强
        'flipud': 0.0,                     # 上下翻转
        'fliplr': 0.5,                     # 左右翻转
        'mosaic': 1.0,                     # 马赛克增强
        'mixup': 0.0,                      # Mixup增强
        'copy_paste': 0.0,                 # 复制粘贴增强
        
        # 训练配置
        'patience': 50,                    # 早停耐心值
        'save_period': 5,                  # 保存间隔
        'val': True,                       # 训练期间验证
        'plots': True,                     # 保存图表
        'cache': 'ram',                    # 缓存到内存
        
        # 输出配置
        'project': 'runs/train',
        'name': 'yolov11_x_gpu', # 修正: 更改实验名称
        'exist_ok': True,
        'verbose': True,
        
        # 高级配置
        'cos_lr': False,                   # 余弦学习率调度
        'close_mosaic': 10,                # 最后10个epoch关闭mosaic
        'resume': False,                   # 从断点恢复
        'overlap_mask': True,              # 重叠掩码
        'mask_ratio': 4,                   # 掩码比例
        'dropout': 0.0,                    # Dropout
        'single_cls': False,               # 多类检测
    }
    
    # 6. 显示配置信息
    print(f"\n⚙️ 训练配置:")
    print(f"   设备: {device}")
    print(f"   批次大小: {batch_size}")
    print(f"   图像尺寸: {training_args['imgsz']}")
    print(f"   训练轮次: {training_args['epochs']}")
    print(f"   混合精度: {amp}")
    print(f"   数据加载线程: {workers}")
    
    # 7. 开始训练
    print(f"\n🏋️ 开始训练...")
    try:
        # 监控GPU内存使用
        if gpu_available:
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 训练模型
        results = model.train(**training_args)
        
        print("✅ 训练完成！")
        
        # 8. 显示训练结果
        print(f"\n📊 训练结果:")
        print(f"   最佳模型: {model.trainer.best}")
        print(f"   最后模型: {model.trainer.last}")
        print(f"   结果目录: {model.trainer.save_dir}")
        
        # 9. 模型验证
        print(f"\n🔍 模型验证...")
        metrics = model.val()
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        # 10. 导出模型（可选）
        export_formats = ['onnx', 'torchscript']
        for fmt in export_formats:
            try:
                print(f"\n📦 导出{fmt.upper()}格式...")
                export_path = model.export(format=fmt, imgsz=640)
                print(f"   {fmt.upper()}模型: {export_path}")
            except Exception as e:
                print(f"   {fmt.upper()}导出失败: {e}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("\n🔧 建议:")
        print("   1. 检查数据集路径是否正确")
        print("   2. 降低批次大小 (batch_size)")
        print("   3. 减少图像尺寸 (imgsz)")
        print("   4. 关闭混合精度训练 (amp=False)")
        
        # GPU内存不足时的处理
        if gpu_available and "out of memory" in str(e).lower():
            print("   💡 GPU内存不足，建议:")
            print("      - 减小batch_size到16或8")
            print("      - 减小imgsz到416")
            print("      - 设置cache=False")
    
    print(f"\n🎉 任务完成！")
    print("=" * 60)
    
    # 11. 清理GPU缓存
    if gpu_available:
        torch.cuda.empty_cache()
        print("🧹 GPU缓存已清理")
    
    # 12. 提供使用建议
    print(f"\n💡 使用建议:")
    print(f"   1. 训练结果保存在: runs/train/yolov11_eais_gpu/")
    print(f"   2. 最佳模型权重: runs/train/yolov11_eais_gpu/weights/best.pt")
    print(f"   3. 训练图表: runs/train/yolov11_eais_gpu/results.png")
    print(f"   4. 可以使用model.predict()进行推理测试")
    
    # 推理示例代码
    print(f"\n📝 推理示例:")
    print(f"""
# 加载训练好的模型
model = YOLO('runs/train/yolov11_eais_gpu/weights/best.pt')

# 进行推理
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
    """)
