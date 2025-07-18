# 🚀 阿里云EAIS部署指南

## 📋 部署准备清单

### 1. 注册魔搭社区账号
- 访问：https://modelscope.cn/
- 注册并实名认证
- 申请EAIS免费实例

### 2. 获取免费GPU实例
- 登录魔搭社区
- 进入"算力中心" -> "弹性计算EAIS"
- 申请免费GPU实例（通常是T4/V100）

## 📦 文件上传清单

### 必须上传的文件：
```
1. 训练脚本：train_yolov11_eais_gpu.py
2. 数据集配置：tiaozhanbei_eais.yaml  
3. 数据集文件：整个 tiaozhanbei_datasets_final 文件夹
4. 预训练模型：yolo11n.pt（可选，会自动下载）
```

### 数据集文件结构（需要保持一致）：
```
tiaozhanbei_datasets_final/
├── images/
│   ├── train/     # 训练图片
│   └── val/       # 验证图片  
└── labels/
    ├── train/     # 训练标签
    └── val/       # 验证标签
```

## 🛠 环境配置命令

### 1. 更新系统
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装依赖
```bash
# 安装Python和pip
sudo apt install python3 python3-pip -y

# 安装深度学习环境
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装YOLOv11
pip3 install ultralytics

# 安装其他依赖
pip3 install opencv-python matplotlib seaborn pandas numpy pillow
```

### 3. 验证GPU环境
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 📁 目录结构设置

### 创建工作目录：
```bash
mkdir -p /workspace/yolo_project
cd /workspace/yolo_project

# 创建必要子目录
mkdir -p datasets configs runs
```

### 上传文件到对应目录：
```bash
/workspace/yolo_project/
├── train_yolov11_eais_gpu.py           # 训练脚本
├── tiaozhanbei_eais.yaml               # 数据集配置
├── datasets/
│   └── tiaozhanbei_datasets_final/     # 数据集
├── configs/
└── runs/                               # 训练结果输出
```

## 🚀 启动训练

### 1. 检查配置
```bash
# 验证数据集路径
ls -la /workspace/datasets/tiaozhanbei_datasets_final/

# 检查YAML配置
cat tiaozhanbei_eais.yaml
```

### 2. 修改配置文件路径
编辑 `tiaozhanbei_eais.yaml`，确保路径正确：
```yaml
path: /workspace/datasets/tiaozhanbei_datasets_final
train: images/train
val: images/val
```

### 3. 启动训练
```bash
# 后台运行训练（推荐）
nohup python3 train_yolov11_eais_gpu.py > training.log 2>&1 &

# 或者直接运行
python3 train_yolov11_eais_gpu.py
```

### 4. 监控训练进度
```bash
# 查看实时日志
tail -f training.log

# 查看GPU使用情况
nvidia-smi

# 查看训练结果
ls -la runs/train/yolov11_eais_gpu/
```

## 📊 性能优化建议

### GPU内存优化：
```python
# 如果遇到GPU内存不足，修改以下参数：
batch_size = 16    # 从32降到16
imgsz = 416       # 从640降到416
cache = False     # 关闭缓存
workers = 4       # 减少数据加载线程
```

### 训练速度优化：
```python
# 启用混合精度训练
amp = True

# 使用更高效的数据加载
cache = 'ram'     # 如果内存充足
workers = 8       # 根据CPU核数调整
```

## 📈 训练监控

### 1. TensorBoard可视化（可选）
```bash
# 安装TensorBoard
pip3 install tensorboard

# 启动TensorBoard
tensorboard --logdir=runs/train --host=0.0.0.0 --port=6006
```

### 2. 结果文件说明
```
runs/train/yolov11_eais_gpu/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最新模型权重
├── results.csv          # 训练指标记录
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── labels.jpg           # 标签分布
├── predictions.jpg      # 预测结果展示
└── args.yaml           # 训练参数记录
```

## 🔧 常见问题解决

### 1. GPU内存不足
```python
# 解决方案：减小批次大小
batch_size = 8  # 进一步减小

# 或者使用梯度累积
accumulate = 4  # 累积4个batch的梯度
```

### 2. 数据加载错误
```bash
# 检查数据集路径
find /workspace -name "*.jpg" | head -5

# 检查标签文件
find /workspace -name "*.txt" | head -5
```

### 3. 网络连接问题
```python
# 使用本地模型文件
model = YOLO('/path/to/local/yolo11n.pt')

# 或者设置镜像源
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ ultralytics
```

## 💾 结果下载

### 训练完成后下载：
```bash
# 打包训练结果
tar -czf yolo_training_results.tar.gz runs/train/yolov11_eais_gpu/

# 下载到本地（使用scp或Web界面）
```

## 🎯 成本优化

### 免费实例使用技巧：
1. **合理安排训练时间**：选择空闲时段
2. **分阶段训练**：可以先训练50epochs，根据效果决定是否继续
3. **及时停止**：达到预期效果后及时停止训练
4. **批量实验**：一次准备多个配置，连续测试

### 实例配置推荐：
- **GPU类型**：T4 (16GB) 或 V100 (32GB)
- **内存**：16GB+ 系统内存
- **存储**：50GB+ SSD存储
- **网络**：稳定的网络连接用于数据上传

## 📝 实验记录模板

```markdown
## 训练实验记录

**实验配置：**
- GPU型号：[T4/V100/A10]
- 批次大小：[16/32/64]  
- 图像尺寸：[416/640/800]
- 训练轮次：[50/100/200]
- 学习率：[0.01/0.001]

**训练结果：**
- 最佳mAP50：[数值]
- 最佳mAP50-95：[数值]
- 训练时长：[小时]
- GPU利用率：[百分比]

**优化建议：**
- [根据结果提出改进建议]
```

这样的配置应该能很好地在阿里云EAIS实例上运行您的YOLOv11训练任务！
