# 基于轻量化国产大模型的高帧频弱小目标检测识别技术研究

本项目致力于探索和实现一个基于YOLOv11的高效、轻量化的目标检测系统，特别针对高帧频场景下的弱小目标进行优化。

## ✨ 功能特性

- **先进的模型架构**: 基于 YOLOv11，兼顾了高精度与高效率。
- **弱小目标优化**: 采用多尺度特征融合等策略，提升对微小目标的检测能力。
- **一键启动**: 提供封装好的 Shell 脚本，简化训练流程，避免参数错误。
- **容器化部署**: 内置 `Dockerfile`，支持一键式环境打包与部署，确保环境一致性。
- **清晰的项目结构**: 代码、配置、文档分离，易于理解和维护。

## 📂 目录结构

```
Yolo-for-GitHub/
├── configs/                # 存放数据集和训练的配置文件
├── docs/                   # 存放项目相关的所有文档
├── eais_deployment/        # 存放EAIS平台部署相关的文件
├── scripts/                # 存放数据预处理等辅助脚本
├── models/                 # 存放预训练模型 (如 yolo11x.pt)
├── Dockerfile              # 用于构建Docker容器的配置文件
├── README.md               # 项目介绍文档
├── requirements.txt        # 项目Python依赖列表
├── start_training.sh       # [核心] 项目唯一推荐的启动脚本
└── train_yolov11.py        # [核心] 项目唯一的训练脚本
```

## 🚀 一键启动训练 (唯一推荐方式)

我们提供了一个 `start_training.sh` 脚本，可以实现一键启动训练，无需手动修改任何Python代码。

### 1. 准备云端环境

在您的云端服务器上，确保您的目录结构满足以下**约定**：

```
/home/user/workspace/  (可以是任何根目录)
├── datasets/
│   └── tiaozhanbei_sequence_split/  <-- 您上传的、已分割好的数据集
│       ├── images/
│       ├── labels/
│       └── tiaozhanbei_seq_data.yaml
│
└── Yolo-for-GitHub/                 <-- 从 Git 克隆的本项目
    ├── models/
    │   └── yolo11x.pt               <-- [必须] 请将 yolo11x.pt 模型放于此处
    ├── scripts/
    ├── configs/
    └── start_training.sh
```

**操作步骤**:
1.  **克隆本项目**:
    ```bash
    git clone https://github.com/Your-Username/Yolo-for-GitHub.git
    ```
2.  **上传数据集**:
    将您本地已处理好的 `tiaozhanbei_sequence_split` 文件夹，整个上传到与 `Yolo-for-GitHub` **并列**的 `datasets` 文件夹中。

3.  **准备预训练模型**:
    将您的预训练权重 `yolo11x.pt` 放入 `Yolo-for-GitHub/models/` 目录下。如果找不到，脚本会尝试自动下载。

### 2. 执行一键启动脚本

进入项目目录，并赋予脚本执行权限，然后运行它。

```bash
cd Yolo-for-GitHub
chmod +x start_training.sh
./start_training.sh
```

脚本会自动寻找数据集配置、检查模型、并启动 `train_yolov11.py` 脚本进行训练。训练结果将保存在 `Yolo-for-GitHub` 外的 `training_runs` 文件夹中，以保持项目目录的干净。

## 🛡️ 如何防止云实例休眠/断线 (重要)

长时间在云端训练时，为防止SSH断开或云平台自动休眠导致训练中断，强烈建议使用 `tmux`。

1.  **安装 tmux** (如果您的服务器没有的话):
    ```bash
    # 如果您是 root 用户 (提示符为 #)
    apt update && apt install -y tmux
    
    # 如果您是普通用户 (提示符为 $)
    sudo apt update && sudo apt install -y tmux
    ```

2.  **创建并进入 tmux 会话**:
    ```bash
    tmux new -s training
    ```

3.  **在 tmux 会话中开始训练**:
    ```bash
    cd /path/to/Yolo-for-GitHub/
    ./start_training.sh
    ```

4.  **安全离开会话**:
    按下组合键 `Ctrl + b`，然后松开，再按 `d`。现在您可以安全关闭终端，训练将在后台继续。

5.  **重连会话查看进度**:
    重新登录服务器后，运行 `tmux attach -t training` 即可。

## 📖 数据集准备

(此部分保持不变，因为 `scripts/create_sequence_split.py` 仍然是准备数据的关键步骤)

**警告**: 为了满足高帧频任务的要求，数据集**必须**按视频序列进行分割，以保证帧的连续性。随机分割帧会破坏时序信息，导致模型性能下降。

本项目提供了一个专用脚本 `scripts/create_sequence_split.py` 来完成此项重要任务。

1.  **准备原始数据集**:
    请确保您的原始数据集包含 `images_diff` 和 `labels_diff` 文件夹，并且在这两个文件夹内部，数据是按序列组织的（例如 `data01`, `data02`, ...）。
    ```
    <your_original_dataset_root>/
    ├── images_diff/
    │   ├── data01/
    │   ├── data02/
    │   └── ...
    └── labels_diff/
        ├── data01/
        ├── data02/
        └── ...
    ```

2.  **运行序列分割脚本**:
    执行以下命令来创建符合训练要求的新数据集。
    ```bash
    python scripts/create_sequence_split.py \
      --source-dir /path/to/your_original_dataset_root \
      --dest-dir /path/to/your_new_dataset_output
    ```
    -   `--source-dir`: 指向您准备好的原始数据集的根目录。
    -   `--dest-dir`: 指定一个新目录，用于存放分割后的数据集。

    脚本执行后，将在 `<dest-dir>` 目录下生成正确的 `images/`、`labels/` 目录结构以及一个 `tiaozhanbei_seq_data.yaml` 配置文件。

### 模型训练

在训练时，请确保使用新脚本生成的 `tiaozhanbei_seq_data.yaml` 配置文件。

**示例：使用GPU进行训练**
```bash
python train_yolov11_gpu_optimized.py \
  --data /path/to/your_new_dataset_output/tiaozhanbei_seq_data.yaml \
  --weights ./path/to/pretrained_model.pt \
  --epochs 100 \
  --batch-size 16 \
  --project ./runs/train \
  --name seq_split_experiment_1
```

### 模型推理

(待补充，需要一个 `detect.py` 脚本)

## 🙏 致谢

本项目的部分研究受到了以下工作的启发... (在此处添加致谢) 