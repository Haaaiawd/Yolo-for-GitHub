#!/bin/bash
# ==============================================================================
# 云端T4环境快速部署脚本
# ==============================================================================
#
# 用于在云端GPU环境中快速设置YOLOv11 5通道训练环境
#
# 使用方法:
#   bash setup_cloud_environment.sh
#
# ==============================================================================

set -e

echo "🚀 开始设置云端T4训练环境..."
echo "=================================="

# 检查Python版本
echo "📋 检查Python环境..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python版本: $python_version"

# 升级pip
echo "📦 升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖
echo "📦 安装Python依赖包..."
pip3 install -r requirements.txt

# 检查CUDA环境
echo "🎮 检查CUDA环境..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 检查ultralytics
echo "🔧 检查Ultralytics安装..."
python3 -c "
from ultralytics import YOLO
print('Ultralytics导入成功')
"

# 设置权限
echo "🔐 设置脚本权限..."
chmod +x start_training.sh
chmod +x check_cloud_environment.py
chmod +x check_optical_flow.py
chmod +x scripts/data_preprocessing/*.py

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p runs/train
mkdir -p runs/val

# 运行环境检查
echo "🔍 运行环境检查..."
python3 check_cloud_environment.py

echo ""
echo "✅ 云端环境设置完成！"
echo "=================================="
echo "下一步操作:"
echo "1. 确保数据集已上传到 ../datasets/ 目录（只需images和labels，不需要flow）"
echo "2. 检查光流状态: python3 check_optical_flow.py"
echo "3. 运行训练: bash start_training.sh（自动计算光流+训练）"
echo "4. 监控训练: tail -f runs/train/*/train.log"
echo "=================================="
