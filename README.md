# 基于轻量化国产大模型的高帧频弱小目标检测识别技术研究

本项目致力于探索和实现一个基于YOLOv11的高效、轻量化的目标检测系统，特别针对高帧频场景下的弱小目标进行优化。项目利用先进的深度学习技术，并适配国产化硬件（如NPU），旨在提供一个可在边缘设备上实时运行的解决方案。

## ✨ 功能特性

- **先进的模型架构**: 基于 YOLOv11，兼顾了高精度与高效率。
- **弱小目标优化**: 采用多尺度特征融合等策略，提升对微小目标的检测能力。
- **硬件加速支持**: 提供了针对不同硬件（CPU, GPU, NPU）的优化训练脚本。
- **容器化部署**: 内置 `Dockerfile`，支持一键式环境打包与部署，确保环境一致性。
- **清晰的项目结构**: 代码、配置、文档分离，易于理解和维护。

## 📂 目录结构

```
Yolo-for-GitHub/
├── configs/                # 存放数据集和训练的配置文件
├── docs/                   # 存放项目相关的所有文档
├── eais_deployment/        # 存放EAIS平台部署相关的文件
├── scripts/                # 存放数据预处理等辅助脚本
├── Dockerfile              # 用于构建Docker容器的配置文件
├── README.md               # 项目介绍文档
├── requirements.txt        # 项目Python依赖列表
└── train_*.py              # 各式训练脚本
```

## 🚀 环境设置

### 方案一: 使用 Docker (推荐)

这是最推荐的方式，可以保证环境的完全一致性。

1.  **安装 Docker**: 请确保您的系统中已安装 [Docker](https://www.docker.com/products/docker-desktop)。

2.  **构建 Docker 镜像**: 在项目根目录下，运行以下命令：
    ```bash
    docker build -t yolo-project .
    ```

3.  **运行 Docker 容器**:
    ```bash
    # -it: 交互式运行
    # --rm: 容器停止后自动删除
    # -v: 将本地的数据集和输出目录挂载到容器中
    docker run -it --rm \
      -v /path/to/your/datasets:/app/datasets \
      -v /path/to/your/outputs:/app/runs \
      yolo-project
    ```
    请将 `/path/to/your/datasets` 和 `/path/to/your/outputs` 替换为您本地的实际路径。

### 方案二: 本地环境手动设置

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/your-username/Yolo-for-GitHub.git
    cd Yolo-for-GitHub
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 如何使用

### 数据集准备 (关键步骤)

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

## ☁️ Cloud Studio 部署指南

1.  **克隆项目**: 在您的 Cloud Studio 工作空间中，通过 `git clone` 将本仓库克隆下来。
2.  **准备数据集**: 使用 Cloud Studio 的文件上传功能，将您的数据集上传到工作空间中（例如，`/home/user/datasets`）。
3.  **安装依赖**: 在终端中运行 `pip install -r requirements.txt`。
4.  **开始训练**:
    ```bash
    python train_yolov11_gpu_optimized.py --data ... (请修改路径以指向您在Cloud Studio中的数据集)
    ```

## 🙏 致谢

本项目的部分研究受到了以下工作的启发... (在此处添加致谢) 