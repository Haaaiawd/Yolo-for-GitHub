# ==============================================================================
# YOLOv11 5通道训练启动脚本 (RGB + 光流)
# ==============================================================================
#
# 基于预训练的 yolo11x.pt 模型进行5通道扩展训练
# 支持RGB图像 + 光流数据的多模态输入
#
# 使用方法:
#   cd Yolo-for-GitHub
#   .\start_5channel_training.ps1
#   或者带参数: .\start_5channel_training.ps1 -epochs 100 -batch 4
#
# ==============================================================================

param(
    [int]$epochs = 50,
    [int]$batch = 2,
    [int]$imgsz = 640,
    [string]$device = "cpu",
    [string]$name = "yolo11x_5channel_optical_flow"
)

Write-Host "🚀 启动YOLOv11 5通道训练 (RGB + 光流)" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "训练参数配置:" -ForegroundColor Yellow
Write-Host "  模型: models/yolo11x.pt (5通道扩展)" -ForegroundColor White
Write-Host "  数据集: configs/tiaozhanbei_final.yaml" -ForegroundColor White
Write-Host "  训练轮数: $epochs" -ForegroundColor White
Write-Host "  批处理大小: $batch" -ForegroundColor White
Write-Host "  图像尺寸: $imgsz" -ForegroundColor White
Write-Host "  设备: $device" -ForegroundColor White
Write-Host "  实验名称: $name" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Magenta

# 检查必要文件
$modelFile = "models\yolo11x.pt"
$configFile = "configs\tiaozhanbei_final.yaml"

if (!(Test-Path $modelFile)) {
    Write-Host "❌ 模型文件不存在: $modelFile" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $configFile)) {
    Write-Host "❌ 配置文件不存在: $configFile" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 所有必要文件检查通过" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 开始5通道训练..." -ForegroundColor Cyan

# 启动训练
python train_yolov11.py `
    --model $modelFile `
    --data $configFile `
    --epochs $epochs `
    --batch $batch `
    --imgsz $imgsz `
    --device $device `
    --name $name `
    --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 训练完成！" -ForegroundColor Green
    Write-Host "📊 结果保存在: runs\train\$name" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ 训练过程中出现错误" -ForegroundColor Red
    Write-Host "请检查上面的错误信息" -ForegroundColor Yellow
}
