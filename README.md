# YOLOv11 5通道训练 - 云端T4部署指南

## 🚀 快速部署

### 1. 环境准备
```bash
# 克隆项目到云端环境
git clone <your-repo-url>
cd Yolo-for-GitHub

# 运行环境设置脚本
bash setup_cloud_environment.sh
```

### 2. 数据集上传
**重要**: 只需上传图像和标签，光流数据将在云端自动计算！

确保数据集结构如下：
```
../datasets/balanced_tiaozhanbei_split/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**注意**:
- ❌ **不需要**上传 `flow/` 目录（文件太大）
- ✅ 光流数据将在云端T4环境中自动计算（速度更快）
- ⏱️ 首次运行时会自动计算光流，大约需要10-30分钟

### 3. 启动训练
```bash
# 检查环境
python3 check_cloud_environment.py

# 开始训练（包含自动光流计算）
bash start_training.sh
```

**首次运行流程**:
1. 🔍 检查数据集结构
2. 🌊 自动计算光流数据（10-30分钟）
3. 📝 生成训练配置文件
4. 🚀 开始5通道模型训练

**后续运行**: 如果光流数据已存在，将直接开始训练

## 📊 T4 GPU优化配置

### 推荐训练参数
- **Batch Size**: 16-32 (根据显存调整)
- **Image Size**: 640 (平衡精度和速度)
- **Epochs**: 100-200
- **Learning Rate**: 自动调整

### 预期性能
- **T4 GPU (16GB)**: 
  - Batch Size 16: ~6-8小时/100 epochs
  - Batch Size 32: ~4-5小时/100 epochs (如果显存足够)

## 🔧 配置文件说明

### 模型配置
- **模型文件**: `models/yolo11x.pt` (预训练模型)
- **输入通道**: 5通道 (RGB + 2通道光流)
- **输出类别**: 6类 (drone, bus, ship, car, cyclist, pedestrian)

### 数据配置
- **云端配置**: `configs/tiaozhanbei_cloud.yaml` (相对路径)
- **本地配置**: `configs/tiaozhanbei_final.yaml` (绝对路径)

## 📈 监控训练

### 实时监控
```bash
# 查看训练日志
tail -f runs/train/yolo11x_5channel_t4_exp1/train.log

# 查看GPU使用情况
nvidia-smi -l 1
```

### 训练结果
训练结果保存在：
- **模型权重**: `runs/train/yolo11x_5channel_t4_exp1/weights/`
- **训练图表**: `runs/train/yolo11x_5channel_t4_exp1/`
- **验证结果**: `runs/train/yolo11x_5channel_t4_exp1/val/`

## 🛠️ 故障排除

### 常见问题

1. **CUDA不可用**
   ```bash
   # 检查CUDA安装
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **显存不足**
   - 减小batch_size: `bash start_training.sh --batch 8`
   - 减小图像尺寸: `bash start_training.sh --imgsz 512`

3. **数据集路径错误**
   - 检查数据集是否在正确位置
   - 确认使用了正确的配置文件

### 性能优化

1. **混合精度训练** (自动启用)
2. **数据加载优化** (多进程)
3. **GPU内存优化** (梯度累积)

## 📝 训练命令参考

```bash
# 基础训练
bash start_training.sh

# 自定义参数训练
bash start_training.sh --epochs 150 --batch 16

# 恢复训练
bash start_training.sh --resume runs/train/yolo11x_5channel_t4_exp1/weights/last.pt

# 调试模式
bash start_training.sh --epochs 1 --batch 1 --imgsz 320
```

## 🎯 预期结果

### 训练指标
- **mAP50**: 目标 >0.7
- **mAP50-95**: 目标 >0.5
- **训练损失**: 应持续下降
- **验证损失**: 应与训练损失趋势一致

### 模型性能
- **推理速度**: ~10-15ms/image (T4 GPU)
- **模型大小**: ~110MB (YOLOv11x)
- **精度**: 预期比3通道模型提升5-10%

## 📞 支持

如遇问题，请检查：
1. 环境检查脚本输出
2. 训练日志文件
3. GPU监控信息
4. 数据集完整性
