#!/bin/bash
# ==============================================================================
# YOLOv11 智能训练启动脚本 (v2.0 - 集成预处理)
# ==============================================================================
#
# 该脚本实现了从数据预处理到启动训练的全流程自动化。
#
# 功能:
# 1. 自动检查最终的数据集、光流和配置文件是否存在。
# 2. 如果不存在，会自动依次执行数据整理、光流计算和配置生成脚本。
# 3. 使用适配光流的5通道 yolo11x 模型启动训练。
#
# 使用方法:
#   cd Yolo-for-GitHub
#   bash start_training.sh [--epochs 150 --batch 8]
#
# ==============================================================================
set -e # 如果任何命令失败，则立即退出脚本

# --- 路径与配置定义 ---
# 基于脚本自身位置的相对路径，确保可移植性
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."
# 修复：脚本在当前项目目录下，不是上级目录
PREPROCESSING_SCRIPTS_DIR="$SCRIPT_DIR/scripts/data_preprocessing"

# 最终需要检查的关键文件和目录
FINAL_DATASET_DIR="$PROJECT_ROOT/datasets/balanced_tiaozhanbei_split"
FINAL_FLOW_DIR="$FINAL_DATASET_DIR/flow"
# 优先使用云端配置，如果不存在则使用本地配置
if [ -f "$SCRIPT_DIR/configs/tiaozhanbei_cloud.yaml" ]; then
    FINAL_CONFIG_FILE="$SCRIPT_DIR/configs/tiaozhanbei_cloud.yaml"
    echo "🌐 使用云端配置文件"
else
    FINAL_CONFIG_FILE="$SCRIPT_DIR/configs/tiaozhanbei_final.yaml"
    echo "💻 使用本地配置文件"
fi

# 模型与训练参数 - 云端T4优化配置
MODEL_FILE="$SCRIPT_DIR/models/yolo11x.pt" # 使用预训练模型进行5通道扩展
DATA_CONFIG="$FINAL_CONFIG_FILE"
EPOCHS=100
BATCH_SIZE=16 # T4 GPU可以支持更大的batch size
IMG_SIZE=640
PROJECT_NAME="runs/train"
EXPERIMENT_NAME="yolo11x_5channel_t4_exp1"
DEVICE="0" # T4 GPU设备

# --- 自动化预处理检查与执行 ---
echo "▶️ [1/3] 正在检查数据集和配置文件..."

# 检查数据集是否存在
if [ ! -d "$FINAL_DATASET_DIR" ]; then
    echo "❌ 错误: 数据集目录不存在: $FINAL_DATASET_DIR"
    echo "请确保数据集已上传到正确位置"
    exit 1
fi

# 检查是否需要预处理
need_preprocessing=false

if [ ! -d "$FINAL_FLOW_DIR" ]; then
    echo "⚠️  光流目录不存在，需要计算光流"
    need_preprocessing=true
fi

if [ ! -f "$FINAL_CONFIG_FILE" ]; then
    echo "⚠️  配置文件不存在，需要生成配置"
    need_preprocessing=true
fi

# 检查光流文件数量
if [ -d "$FINAL_FLOW_DIR" ]; then
    flow_count=$(find "$FINAL_FLOW_DIR" -name "*.npy" | wc -l)
    echo "📊 现有光流文件数量: $flow_count"
    if [ "$flow_count" -lt 100 ]; then
        echo "⚠️  光流文件数量不足，需要重新计算"
        need_preprocessing=true
    fi
fi

if [ "$need_preprocessing" = true ]; then
    echo "🚀 启动云端预处理流程..."
    echo "⚡ 在云端GPU环境中计算光流将显著加速处理..."

    # 依次执行预处理脚本
    echo "  ▶️ [Step A] 检查文件结构..."
    echo "     脚本路径: $PREPROCESSING_SCRIPTS_DIR/organize_files.py"
    if [ -f "$PREPROCESSING_SCRIPTS_DIR/organize_files.py" ]; then
        python "$PREPROCESSING_SCRIPTS_DIR/organize_files.py"
        echo "  ✅ 文件结构检查完成。"
    else
        echo "  ❌ 错误: 找不到organize_files.py脚本"
        echo "     期望路径: $PREPROCESSING_SCRIPTS_DIR/organize_files.py"
        echo "     当前目录: $(pwd)"
        echo "     脚本目录内容:"
        ls -la "$PREPROCESSING_SCRIPTS_DIR/" || echo "     目录不存在"
        exit 1
    fi

    echo "  ▶️ [Step B] 🌊 批量计算光流 (云端GPU加速)..."
    echo "     这可能需要10-30分钟，请耐心等待..."
    start_time=$(date +%s)
    if [ -f "$PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py" ]; then
        python "$PREPROCESSING_SCRIPTS_DIR/compute_optical_flow.py"
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ✅ 光流计算完成，耗时: ${duration}秒 ($((duration/60))分钟)"
    else
        echo "  ❌ 错误: 找不到compute_optical_flow.py脚本"
        exit 1
    fi

    echo "  ▶️ [Step C] 创建云端配置文件..."
    if [ -f "$PREPROCESSING_SCRIPTS_DIR/create_final_yaml.py" ]; then
        python "$PREPROCESSING_SCRIPTS_DIR/create_final_yaml.py"
        echo "  ✅ 配置文件创建完成。"
    else
        echo "  ❌ 错误: 找不到create_final_yaml.py脚本"
        exit 1
    fi

    echo "🎉 云端预处理流程执行完毕！"
else
    echo "✅ 数据集和配置文件均已就绪，跳过预处理。"
fi

# --- 启动训练 ---
echo "▶️ [2/3] 准备启动训练..."

# 解析命令行传入的额外参数
EXTRA_ARGS="$@"

echo "========================================="
echo "  YOLOv11 5-Channel Training on T4 GPU  "
echo "========================================="
echo "-> Model File:      $MODEL_FILE"
echo "-> Dataset Config:  $DATA_CONFIG"
echo "-> Epochs:          $EPOCHS"
echo "-> Batch Size:      $BATCH_SIZE"
echo "-> Image Size:      $IMG_SIZE"
echo "-> Project:         $PROJECT_NAME"
echo "-> Experiment:      $EXPERIMENT_NAME"
echo "-> Device:          $DEVICE (T4 GPU)"
echo "-> Extra Args:      $EXTRA_ARGS"
echo "-----------------------------------------"

# 切换到脚本所在目录执行，确保相对路径正确
cd "$SCRIPT_DIR"

echo "▶️ [3/3] 执行训练命令..."
echo "🚀 启动5通道训练 (RGB + 光流) on T4 GPU..."

# 检查GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"

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