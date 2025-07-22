#!/bin/bash
# ==============================================================================
# YOLOv11 训练启动脚本 (v2.2 - 集成光流检查)
# ==============================================================================
#
# 该脚本在启动训练前会自动检查光流文件是否存在。
# 如果光流不存在，它会自动运行计算脚本。
#
# 使用方法:
#   bash start_training.sh [--epochs 150 --batch 8]
#
# ==============================================================================
set -e # 如果任何命令失败，则立即退出脚本

# --- 路径与配置定义 ---
# 基于脚本自身位置的相对路径，确保可移植性
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- 固定使用的配置文件和模型 ---
DATA_CONFIG="$SCRIPT_DIR/configs/tiaozhanbei_cloud.yaml"
MODEL_FILE="$SCRIPT_DIR/models/yolo11x.pt" 
PREPROCESSING_SCRIPTS_DIR="$SCRIPT_DIR/scripts/data_preprocessing"
# 假设的数据集路径，光流计算脚本可能需要
FINAL_DATASET_DIR="$SCRIPT_DIR/datasets/balanced_tiaozhanbei_split"
FINAL_FLOW_DIR="$FINAL_DATASET_DIR/flow"

# --- 训练参数 ---
# 您可以在这里修改默认值，或通过命令行参数覆盖
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
PROJECT_NAME="runs/train"
EXPERIMENT_NAME="yolo11x_5channel_exp"
DEVICE="0" # 默认使用第一个GPU

# --- 自动化光流检查与计算 ---
echo "▶️ [1/4] 正在检查光流文件..."
if [ ! -d "$FINAL_FLOW_DIR" ]; then
    echo "⚠️  光流目录不存在: $FINAL_FLOW_DIR"
    echo "🚀 将自动执行光流计算脚本..."
    
    if [ -f "$PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py" ]; then
        echo "   执行脚本: $PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py"
        python "$PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py"
        echo "✅ 光流计算完成。"
    else
        echo "❌ 错误: 找不到 compute_optical_flow.py 脚本"
        echo "   请确保脚本位于: $PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py"
        exit 1
    fi
else
    echo "✅ 光流目录已找到。"
fi

# --- 启动前检查 ---
echo "▶️ [2/4] 正在检查必需文件..."

if [ ! -f "$DATA_CONFIG" ]; then
    echo "❌ 错误: 找不到数据配置文件: $DATA_CONFIG"
    echo "请确保该文件存在于 configs/ 目录下。"
    exit 1
fi
echo "✅ 数据配置文件已找到: $DATA_CONFIG"

if [ ! -f "$MODEL_FILE" ]; then
    echo "⚠️  警告: 找不到预训练模型文件: $MODEL_FILE"
    echo "   训练将从头开始，或者由训练脚本尝试自动下载。"
else
    echo "✅ 预训练模型已找到: $MODEL_FILE"
fi

# --- 准备启动训练 ---
echo "▶️ [3/4] 准备启动训练..."

# 解析命令行传入的额外参数以覆盖默认值
EXTRA_ARGS="$@"

echo "========================================="
echo "    YOLOv11 5-Channel Training           "
echo "========================================="
echo "-> Model File:      $MODEL_FILE"
echo "-> Dataset Config:  $DATA_CONFIG"
echo "-> Epochs:          $EPOCHS"
echo "-> Batch Size:      $BATCH_SIZE"
echo "-> Image Size:      $IMG_SIZE"
echo "-> Project:         $PROJECT_NAME"
echo "-> Experiment:      $EXPERIMENT_NAME"
echo "-> Device:          $DEVICE"
echo "-> Extra Args:      $EXTRA_ARGS"
echo "-----------------------------------------"

# 切换到脚本所在目录执行，确保相对路径正确
cd "$SCRIPT_DIR"

echo "▶️ [4/4] 执行训练命令..."

# 检查GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"

# 执行训练
python train_yolov11.py \
    --model "$MODEL_FILE" \
    --data "$DATA_CONFIG" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --project "$PROJECT_NAME" \
    --name "$EXPERIMENT_NAME" \
    --device "$DEVICE" \
    $EXTRA_ARGS

echo "========================================="
echo "  训练脚本执行结束。"
echo "=========================================" 