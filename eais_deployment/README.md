# 🚀 YOLOv11 阿里云EAIS 一键部署和训练

这是一个针对阿里云弹性加速计算服务 (EAIS) 优化的 YOLOv11 自动化部署和训练解决方案。

## ✨ 特性

- **🎯 一键启动**: 只需运行一个 Jupyter 单元格，即可自动完成所有部署和训练步骤
- **🛡️ 防断线**: 训练在后台运行，不受 SSH 断开或网络中断影响
- **📊 智能监控**: 实时日志输出和训练状态监控
- **🔧 自动配置**: 动态生成正确的数据集配置文件
- **💾 完整结果**: 自动保存最佳模型、训练图表和评估结果

## 📋 使用步骤

### 1. 准备文件

确保您的项目结构如下：
```
project_root/
├── eais_deployment/
│   ├── run_eais_training.sh           # 🆕 一键启动脚本
│   ├── train_yolov11_eais_gpu.py      # 训练脚本
│   ├── tiaozhanbei_eais.yaml          # 数据集配置
│   ├── requirements.txt               # Python依赖
│   └── setup_environment.sh           # 环境安装脚本
├── datasets/
│   └── tiaozhanbei_datasets_final.zip # 数据集压缩包
├── EAIS_文件解压部署.ipynb            # 🆕 一键启动 Notebook
└── 其他项目文件...
```

### 2. 上传到 EAIS 实例

将整个项目文件夹上传到您的阿里云 EAIS 实例。

### 3. 一键启动训练

1. 打开 `EAIS_文件解压部署.ipynb`
2. 运行第一个 Python 代码单元格
3. 等待自动化脚本完成所有设置
4. 训练将在后台自动开始

就这么简单！🎉

## 📊 监控训练

### 实时日志
```bash
tail -f logs/training_*.log
```

### GPU 状态
```bash
nvidia-smi
```

### 训练进程
```bash
ps aux | grep train_yolov11
```

## 📁 训练结果

训练完成后，结果将保存在：
- **最佳模型**: `runs/train/yolov11_eais_gpu/weights/best.pt`
- **训练图表**: `runs/train/yolov11_eais_gpu/results.png`
- **混淆矩阵**: `runs/train/yolov11_eais_gpu/confusion_matrix.png`

## 🔧 技术细节

### 自动化流程

`run_eais_training.sh` 脚本会自动执行：

1. **环境安装**: 安装 PyTorch CUDA 版本和所有依赖
2. **数据集处理**: 自动查找并解压数据集
3. **模型下载**: 下载 YOLOv11n 预训练模型
4. **配置生成**: 创建带有正确绝对路径的 YAML 配置
5. **后台训练**: 使用 nohup 启动训练，防止断线

### 训练配置

- **模型**: YOLOv11 nano (2.59M 参数)
- **数据集**: 6类红外小目标检测
- **训练轮次**: 100 epochs
- **批次大小**: 32 (GPU) / 8 (CPU降级)
- **优化器**: AdamW
- **混合精度**: 启用 (GPU环境)

## 🚨 故障排除

### 常见问题

1. **数据集未找到**
   - 确保 `datasets/tiaozhanbei_datasets_final.zip` 存在
   - 检查文件路径和权限

2. **GPU 内存不足**
   - 脚本会自动降级到较小的批次大小
   - 也可以手动调整 `train_yolov11_eais_gpu.py` 中的参数

3. **网络下载失败**
   - 预训练模型下载失败时，脚本会继续执行
   - Ultralytics 会在训练时自动重试下载

### 手动执行

如果 Jupyter 方式遇到问题，也可以直接在终端执行：
```bash
chmod +x eais_deployment/run_eais_training.sh
bash eais_deployment/run_eais_training.sh
```

## 📈 性能预期

- **训练时间**: 6-12 小时 (取决于 GPU 型号)
- **目标 mAP50**: 0.4-0.6+
- **目标 mAP50-95**: 0.2-0.35+

## 📞 支持

如果遇到问题，请检查：
1. `logs/training_*.log` 中的详细日志
2. GPU 内存和使用情况
3. 数据集文件完整性

---

**🎯 一键启动，专业训练，就是这么简单！**
