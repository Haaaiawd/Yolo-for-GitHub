#!/bin/bash
set -e # 任何命令失败时立即退出

#
# 🚀 YOLOv11 阿里云EAIS 一键训练启动脚本
#
# 这个脚本将自动化完成以下所有步骤:
# 1. 环境安装: 安装所有必要的Python依赖和PyTorch
# 2. 数据集准备: 自动查找并解压 tiaozhanbei_datasets_final.zip
# 3. 模型下载: 下载yolo11n.pt预训练模型
# 4. 配置文件生成: 创建指向正确数据集路径的yaml文件
# 5. 后台训练启动: 使用nohup启动训练，防止终端断开
#

# --- 配置 ---
DATASET_ZIP_FILE="datasets/tiaozhanbei_datasets_final.zip"
DATASET_EXTRACT_PATH="datasets"
DATASET_FINAL_DIR="datasets/tiaozhanbei_datasets_final"
CONFIG_FILE="eais_deployment/tiaozhanbei_eais.yaml"
TRAIN_SCRIPT="eais_deployment/train_yolov11_eais_gpu.py"
PRETRAINED_MODEL="yolo11n.pt"
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
REQUIREMENTS_FILE="eais_deployment/requirements.txt"

# --- 函数 ---

# 打印信息
log_info() {
    echo "✅ [INFO] $1"
}

log_warning() {
    echo "⚠️ [WARNING] $1"
}

log_error() {
    echo "❌ [ERROR] $1"
    exit 1
}

# 1. 环境安装
setup_environment() {
    log_info "开始安装环境依赖..."
    
    log_info "正在安装PyTorch (CUDA 11.8)..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    log_info "正在安装 requirements.txt 中的依赖..."
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip3 install -r "$REQUIREMENTS_FILE"
    else
        log_error "依赖文件 $REQUIREMENTS_FILE 未找到！"
    fi
    
    log_info "验证关键库 'ultralytics' 是否安装成功..."
    python3 -c "from ultralytics import YOLO; print('✅ ultralytics 库导入成功')"
    
    log_info "环境安装完成。"
}

# 2. 数据集准备
prepare_dataset() {
    log_info "开始准备数据集..."
    if [ -d "$DATASET_FINAL_DIR" ]; then
        log_info "数据集目录已存在，跳过解压。"
    elif [ -f "$DATASET_ZIP_FILE" ]; then
        log_info "找到数据集压缩包，开始解压..."
        unzip -q "$DATASET_ZIP_FILE" -d "$DATASET_EXTRACT_PATH"
        log_info "数据集解压完成。"
    else
        log_error "未找到数据集压缩包: $DATASET_ZIP_FILE"
    fi
}

# 3. 模型下载
download_model() {
    log_info "开始准备预训练模型..."
    if [ -f "$PRETRAINED_MODEL" ]; then
        log_info "预训练模型 $PRETRAINED_MODEL 已存在。"
    else
        log_info "下载预训练模型 $PRETRAINED_MODEL..."
        wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O "$PRETRAINED_MODEL"
        if [ $? -eq 0 ]; then
            log_info "模型下载成功。"
        else
            log_error "模型下载失败。"
        fi
    fi
}

# 4. 配置文件生成
create_config() {
    log_info "开始生成配置文件..."
    DATASET_ABS_PATH=$(realpath "$DATASET_FINAL_DIR")

    # 创建YAML配置文件
    cat > "$CONFIG_FILE" <<EOL
# YOLOv11 Dataset configuration for 挑战杯红外小目标检测
path: $DATASET_ABS_PATH
train: images/train
val: images/val
test:

nc: 6

names:
  0: drone
  1: car  
  2: ship
  3: bus
  4: pedestrian
  5: cyclist
EOL
    log_info "配置文件 $CONFIG_FILE 生成成功。"
    log_info "数据集路径设置为: $DATASET_ABS_PATH"
}

# 5. 启动训练
start_training() {
    log_info "开始启动后台训练..."
    mkdir -p logs

    nohup python3 "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
    
    TRAIN_PID=$!
    log_info "训练已在后台启动，进程ID: $TRAIN_PID"
    log_info "日志文件: $LOG_FILE"
    log_info "您可以使用 'tail -f $LOG_FILE' 命令查看实时日志。"
    log_info "您现在可以安全地关闭终端。"
}


# --- 主流程 ---
main() {
    setup_environment
    prepare_dataset
    download_model
    create_config
    start_training
    log_info "🎉 所有任务已启动！"
}

# 执行主流程
main 