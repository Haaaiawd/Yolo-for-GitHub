#!/bin/bash

#-------------------------------------------------------------------------------
# YOLOv11 一键恢复训练脚本 (Pro版 - 自动查找最新)
#-------------------------------------------------------------------------------
# 使用此脚本从最新的训练任务（last.pt）继续训练。
# 脚本会自动查找 ../training_runs 目录下最新的文件夹作为恢复目标。
#
# !! 无需手动配置, 脚本会自动查找 !!
#-------------------------------------------------------------------------------

# --- 用户配置区域 (如果需要,可以修改) ---
OUTPUT_DIR="../training_runs" # 所有训练任务的根目录
DATA_CONFIG_PATH="../datasets/tiaozhanbei_sequence_split/tiaozhanbei_seq_data.yaml"
# ------------------------------------

echo "🚀 开始执行一键恢复训练脚本 (Pro版)..."

# 1. 自动查找最新的实验文件夹
echo "📂 正在扫描 ${OUTPUT_DIR} 目录以查找最新的训练任务..."
# 使用 ls -td 命令列出所有子目录,按修改时间排序,然后取第一个
LATEST_EXPERIMENT_DIR=$(ls -td ${OUTPUT_DIR}/*/ | head -n 1)

if [ -z "$LATEST_EXPERIMENT_DIR" ]; then
    echo "❌ 错误：在 ${OUTPUT_DIR} 中没有找到任何训练任务文件夹。"
    exit 1
fi

EXPERIMENT_NAME=$(basename "$LATEST_EXPERIMENT_DIR")
echo "✅ 找到最新的训练任务: ${EXPERIMENT_NAME}"

# 2. 构建 last.pt 的路径
LAST_PT_PATH="${LATEST_EXPERIMENT_DIR}weights/last.pt"

# 3. 检查关键文件是否存在
if [ ! -f "$LAST_PT_PATH" ]; then
    echo "❌ 错误：在最新的训练任务中找不到 last.pt 模型！"
    echo "请检查路径：${LAST_PT_PATH}"
    exit 1
fi
if [ ! -f "$DATA_CONFIG_PATH" ]; then
    echo "❌ 错误：找不到数据集配置文件！"
    echo "请检查路径：${DATA_CONFIG_PATH}"
    exit 1
fi

echo "✅ 关键文件检查通过。"
echo "📂 训练结果将继续保存在: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "-------------------------------------------------"
echo "🔥 将从 ${LAST_PT_PATH} 恢复训练..."

# 4. 组装并执行恢复训练的命令
# 核心修正：使用 --resume 参数来真正地从断点恢复
python train_yolov11.py \
  --data "$DATA_CONFIG_PATH" \
  --model "$LAST_PT_PATH" \
  --resume "$LAST_PT_PATH" \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --project "$OUTPUT_DIR" \
  --name "$EXPERIMENT_NAME" \
  --device 0 \
  --save_period 5

echo "✅ 训练命令执行完毕。" 