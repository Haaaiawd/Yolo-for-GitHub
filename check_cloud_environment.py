#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端环境检查脚本
用于验证T4 GPU环境是否准备就绪进行5通道YOLO训练
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("   ❌ Python版本过低，需要>=3.8")
        return False

def check_gpu_availability():
    """检查GPU可用性"""
    print("\n🎮 检查GPU环境...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA可用: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"   GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            print("   ✅ GPU环境正常")
            return True
        else:
            print("   ❌ CUDA不可用，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_required_packages():
    """检查必要的Python包"""
    print("\n📦 检查必要的Python包...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'opencv-python',
        'numpy',
        'yaml',
        'tqdm',
        'matplotlib',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        # 特殊处理一些包名
        import_name = package
        if package == 'opencv-python':
            import_name = 'cv2'
        elif package == 'yaml':
            import_name = 'yaml'
            
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} (未找到)")
                missing_packages.append(package)
        except ImportError:
            print(f"   ❌ {package} (导入错误)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   缺失的包: {', '.join(missing_packages)}")
        print("   请运行: pip install " + " ".join(missing_packages))
        return False
    else:
        print("   ✅ 所有必要包已安装")
        return True

def check_project_files():
    """检查项目文件完整性"""
    print("\n📁 检查项目文件...")

    required_files = [
        'train_yolov11.py',
        'models/yolo11x.pt',
        'start_training.sh',
        'scripts/data_preprocessing/compute_optical_flow.py',
        'scripts/data_preprocessing/organize_files.py',
        'scripts/data_preprocessing/create_final_yaml.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (缺失)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n   缺失的文件: {', '.join(missing_files)}")
        return False
    else:
        print("   ✅ 所有必要文件存在")
        return True

def check_dataset():
    """检查数据集"""
    print("\n📊 检查数据集...")

    # 检查基础数据集结构
    dataset_paths = [
        '../datasets/balanced_tiaozhanbei_split/images/train',
        '../datasets/balanced_tiaozhanbei_split/images/val',
        '../datasets/balanced_tiaozhanbei_split/labels/train',
        '../datasets/balanced_tiaozhanbei_split/labels/val'
    ]

    # 光流目录（可能不存在，会在训练时自动生成）
    flow_paths = [
        '../datasets/balanced_tiaozhanbei_split/flow/train',
        '../datasets/balanced_tiaozhanbei_split/flow/val'
    ]
    
    dataset_ok = True

    # 检查基础数据集
    for path in dataset_paths:
        if os.path.exists(path):
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"   ✅ {path} ({file_count} 文件)")
        else:
            print(f"   ❌ {path} (不存在)")
            dataset_ok = False

    # 检查光流数据（可选）
    flow_exists = True
    total_flow_files = 0
    for path in flow_paths:
        if os.path.exists(path):
            flow_count = len([f for f in os.listdir(path) if f.endswith('.npy')])
            total_flow_files += flow_count
            print(f"   ✅ {path} ({flow_count} 光流文件)")
        else:
            print(f"   ⚠️  {path} (不存在，将在训练时自动生成)")
            flow_exists = False

    if not flow_exists:
        print("   💡 光流数据将在首次训练时自动计算")
        print("   ⏱️  预计光流计算时间: 10-30分钟（取决于数据集大小）")
    else:
        print(f"   🌊 光流数据已存在，共 {total_flow_files} 个文件")

    return dataset_ok

def estimate_training_time():
    """估算训练时间"""
    print("\n⏱️  训练时间估算...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if 'T4' in gpu_name:
                print("   T4 GPU环境:")
                print("   - 100 epochs, batch_size=16: 约6-8小时")
                print("   - 50 epochs, batch_size=16: 约3-4小时")
                print("   - 建议使用batch_size=16-32进行训练")
            else:
                print(f"   {gpu_name} GPU环境:")
                print("   - 训练时间取决于具体GPU性能")
        else:
            print("   CPU环境:")
            print("   - 训练时间会显著增加（不推荐）")
    except:
        print("   无法估算训练时间")

def main():
    """主检查函数"""
    print("=" * 50)
    print("🚀 YOLOv11 5通道训练环境检查")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_gpu_availability(), 
        check_required_packages(),
        check_project_files(),
        check_dataset()
    ]
    
    estimate_training_time()
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 环境检查通过！可以开始训练")
        print("运行命令: bash start_training.sh")
    else:
        print("❌ 环境检查失败，请解决上述问题后重试")
    print("=" * 50)

if __name__ == "__main__":
    main()
