#!/bin/bash

#-------------------------------------------------------------------------------
# YOLOv11 一键启动训练脚本
#-------------------------------------------------------------------------------
# 该脚本会自动查找所需文件并启动训练，无需手动修改任何路径。
#
# 使用前，请确保您的云端目录结构如下：
#
# /some_root_folder/
# ├── datasets/
# │   └── tiaozhanbei_sequence_split/  <-- 您上传的、已分割好的数据集
# │       ├── images/
# │       ├── labels/
# │       └── tiaozhanbei_seq_data.yaml
# │
# └── Yolo-for-GitHub/                 <-- 从 Git 克隆的项目 (此脚本所在的位置)
#     ├── scripts/
#     ├── configs/
#     └── start_training.sh
#
#-------------------------------------------------------------------------------

echo "🚀 开始执行一键启动训练脚本..."

# 1. 定义基于约定目录结构的相对路径
#    - 数据配置文件位于项目文件夹的上一级的 `datasets` 目录中
#    - 训练结果也将保存在上一级目录，以避免污染 Git 仓库
DATA_CONFIG_PATH="../datasets/tiaozhanbei_sequence_split/tiaozhanbei_seq_data.yaml"
OUTPUT_DIR="../training_runs"
PRETRAINED_MODEL="./models/yolov8n.pt" # 假设预训练模型在 'models' 文件夹

# 2. 检查最关键的数据集配置文件是否存在
if [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "❌ 错误：找不到数据集配置文件！"
    echo "请确保您的目录结构符合脚本开头的说明。"
    echo "需要将已处理好的 'tiaozhanbei_sequence_split' 文件夹上传到 'Yolo-for-GitHub' 文件夹的旁边。"
    exit 1
fi

echo "✅ 数据集配置文件已找到: ${DATA_CONFIG_PATH}"

# 3. 创建预训练模型目录并检查模型文件 (如果需要)
#    (此处可以添加从网络下载预训练模型的逻辑，例如 `wget ...`)
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "⚠️ 警告：找不到预训练模型 ${PRETRAINED_MODEL}。"
    echo "将尝试从头开始训练。如果需要预训练权重，请将其放置在项目的 'models' 目录下。"
    PRETRAINED_MODEL="" # 清空路径，让训练脚本从头开始
else
    echo "✅ 预训练模型已找到: ${PRETRAINED_MODEL}"
fi

# 4. 组装并执行最终的训练命令
echo "📂 训练结果将保存在: ${OUTPUT_DIR}"
echo "-------------------------------------------------"
echo "🔥 开始训练..."

python train_yolov11_gpu_optimized.py \
  --data "$DATA_CONFIG_PATH" \
  --weights "$PRETRAINED_MODEL" \
  --epochs 100 \
  --batch-size 16 \
  --project "$OUTPUT_DIR" \
  --name "cloud_training_$(date +%Y%m%d_%H%M%S)" \
  --device 0

echo "✅ 训练命令执行完毕。" 