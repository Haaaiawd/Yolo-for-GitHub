#!/bin/bash
# ==============================================================================
# YOLOv11 5通道训练启动脚本 - 简化云端版本
# ==============================================================================

set -e

echo "🚀 YOLOv11 5通道训练启动 (简化版本)"
echo "=================================="

# 检查当前目录
echo "当前目录: $(pwd)"

# 检查必要文件
echo "🔍 检查必要文件..."

required_files=(
    "models/yolo11x.pt"
    "train_yolov11.py"
    "scripts/data_preprocessing/organize_files.py"
    "scripts/data_preprocessing/compute_optical_flow.py"
    "scripts/data_preprocessing/create_final_yaml.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 不存在"
        exit 1
    fi
done

# 检查数据集
echo ""
echo "🔍 检查数据集..."
dataset_dir="../datasets/balanced_tiaozhanbei_split"

if [ ! -d "$dataset_dir" ]; then
    echo "❌ 数据集目录不存在: $dataset_dir"
    echo "请确保数据集已上传到正确位置"
    exit 1
fi

echo "✅ 数据集目录存在"

# 检查光流数据
flow_dir="$dataset_dir/flow"
need_flow_computation=false

if [ ! -d "$flow_dir" ]; then
    echo "⚠️  光流目录不存在，需要计算光流"
    need_flow_computation=true
else
    flow_count=$(find "$flow_dir" -name "*.npy" 2>/dev/null | wc -l)
    echo "📊 现有光流文件数量: $flow_count"
    if [ "$flow_count" -lt 100 ]; then
        echo "⚠️  光流文件数量不足，需要重新计算"
        need_flow_computation=true
    fi
fi

# 执行预处理
if [ "$need_flow_computation" = true ]; then
    echo ""
    echo "🚀 开始预处理流程..."
    
    echo "  ▶️ [1/3] 检查文件结构..."
    python scripts/data_preprocessing/organize_files.py
    echo "  ✅ 文件结构检查完成"
    
    echo "  ▶️ [2/3] 计算光流数据..."
    echo "     这可能需要10-30分钟，请耐心等待..."
    start_time=$(date +%s)
    python scripts/data_preprocessing/compute_optical_flow.py
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "  ✅ 光流计算完成，耗时: ${duration}秒"
    
    echo "  ▶️ [3/3] 生成配置文件..."
    python scripts/data_preprocessing/create_final_yaml.py
    echo "  ✅ 配置文件生成完成"
    
    echo "🎉 预处理完成！"
else
    echo "✅ 光流数据已存在，跳过预处理"
fi

# 确定配置文件
if [ -f "configs/tiaozhanbei_cloud.yaml" ]; then
    config_file="configs/tiaozhanbei_cloud.yaml"
    echo "🌐 使用云端配置文件"
else
    config_file="configs/tiaozhanbei_final.yaml"
    echo "💻 使用本地配置文件"
fi

# 检查GPU
echo ""
echo "🎮 检查GPU环境..."
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
else:
    print('将使用CPU训练（速度较慢）')
"

# 开始训练
echo ""
echo "🚀 开始5通道训练..."
echo "=================================="
echo "模型: models/yolo11x.pt (5通道扩展)"
echo "配置: $config_file"
echo "=================================="

python train_yolov11.py \
    --model models/yolo11x.pt \
    --data "$config_file" \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --project runs/train \
    --name yolo11x_5channel_cloud \
    --device 0 \
    --verbose

echo ""
echo "🎉 训练完成！"
echo "结果保存在: runs/train/yolo11x_5channel_cloud/"
