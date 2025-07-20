# ==============================================================================
# YOLOv11 智能训练启动脚本 (Windows PowerShell版本)
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
#   .\start_training.ps1
#   或者带参数: .\start_training.ps1 -epochs 150 -batch 8
#
# ==============================================================================

param(
    [int]$epochs = 100,
    [int]$batch = 8,
    [int]$imgsz = 640,
    [string]$device = "cpu",
    [string]$name = "yolo11x_5channel_exp1"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# --- 路径与配置定义 ---
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$PREPROCESSING_SCRIPTS_DIR = Join-Path $PROJECT_ROOT "scripts\data_preprocessing"

# 最终需要检查的关键文件和目录
$FINAL_DATASET_DIR = Join-Path $PROJECT_ROOT "datasets\balanced_tiaozhanbei_split"
$FINAL_FLOW_DIR = Join-Path $FINAL_DATASET_DIR "flow"
$FINAL_CONFIG_FILE = Join-Path $SCRIPT_DIR "configs\tiaozhanbei_final.yaml"

# 模型与训练参数
$MODEL_CONFIG = Join-Path $SCRIPT_DIR "configs\yolo11-5ch.yaml"
$DATA_CONFIG = $FINAL_CONFIG_FILE
$PROJECT_NAME = "runs/train"

Write-Host "▶️ [1/3] 正在检查数据集和配置文件..." -ForegroundColor Green

if (!(Test-Path $FINAL_FLOW_DIR) -or !(Test-Path $FINAL_CONFIG_FILE)) {
    Write-Host "⚠️  检测到部分或全部预处理产物缺失，将启动全自动预处理流程..." -ForegroundColor Yellow

    # 依次执行预处理脚本
    Write-Host "  ▶️ [Step A] 按清单整理文件..." -ForegroundColor Cyan
    python (Join-Path $PREPROCESSING_SCRIPTS_DIR "organize_files.py")
    Write-Host "  ✅ 文件整理完成。" -ForegroundColor Green

    Write-Host "  ▶️ [Step B] 批量计算光流 (这可能需要较长时间)..." -ForegroundColor Cyan
    python (Join-Path $PREPROCESSING_SCRIPTS_DIR "compute_optical_flow.py")
    Write-Host "  ✅ 光流计算完成。" -ForegroundColor Green

    Write-Host "  ▶️ [Step C] 创建最终配置文件..." -ForegroundColor Cyan
    python (Join-Path $PREPROCESSING_SCRIPTS_DIR "create_final_yaml.py")
    Write-Host "  ✅ 配置文件创建完成。" -ForegroundColor Green

    Write-Host "✅ 全自动预处理流程执行完毕！" -ForegroundColor Green
} else {
    Write-Host "✅ 数据集和配置文件均已就绪。" -ForegroundColor Green
}

# --- 启动训练 ---
Write-Host "▶️ [2/3] 准备启动训练..." -ForegroundColor Green

Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "  YOLOv11 5-Channel Training Starting  " -ForegroundColor Magenta
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "-> Model Config:    $MODEL_CONFIG" -ForegroundColor White
Write-Host "-> Dataset Config:  $DATA_CONFIG" -ForegroundColor White
Write-Host "-> Epochs:          $epochs" -ForegroundColor White
Write-Host "-> Batch Size:      $batch" -ForegroundColor White
Write-Host "-> Image Size:      $imgsz" -ForegroundColor White
Write-Host "-> Project:         $PROJECT_NAME" -ForegroundColor White
Write-Host "-> Experiment:      $name" -ForegroundColor White
Write-Host "-> Device:          $device" -ForegroundColor White
Write-Host "-----------------------------------------" -ForegroundColor Magenta

# 切换到脚本所在目录执行，确保相对路径正确
Set-Location $SCRIPT_DIR

Write-Host "▶️ [3/3] 执行训练命令..." -ForegroundColor Green

# 检查是否使用5通道模型
if (Test-Path $MODEL_CONFIG) {
    Write-Host "使用5通道模型配置..." -ForegroundColor Yellow
    python train_yolov11.py --model $MODEL_CONFIG --data $DATA_CONFIG --epochs $epochs --batch $batch --imgsz $imgsz --project $PROJECT_NAME --name $name --device $device
} else {
    Write-Host "5通道模型配置不存在，使用标准3通道模型..." -ForegroundColor Yellow
    python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.train(data='$DATA_CONFIG', epochs=$epochs, batch=$batch, imgsz=$imgsz, project='$PROJECT_NAME', name='$name', device='$device')"
}

Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "  训练脚本执行结束。" -ForegroundColor Magenta
Write-Host "=========================================" -ForegroundColor Magenta
