# Technical Specification: 基于YOLOv11的弱小目标检测

**最后更新日期:** 2024-10-16

## 1. 系统架构与技术栈

### 1.1 技术栈 (Technology Stack)
- **编程语言**: Python 3.8+
- **核心框架**: PyTorch 1.12+
- **核心模型库**: Ultralytics 8.3+ (包含YOLOv11)
- **依赖包**:
    - `numpy`
    - `opencv-python`
    - `pyyaml`
    - `matplotlib`
    - `pandas` (用于处理和分析结果)
    - `tqdm` (用于显示进度条)
- **包管理工具**: `pnpm` (根据用户偏好) [[memory:3339367]]
- **硬件环境**:
    - **CPU**: 可在无GPU环境下运行，但速度较慢。
    - **GPU**: 强烈推荐使用NVIDIA GPU以获得理想的训练和推理速度。
    - **特定硬件加速**: 考虑到用户的硬件配置 [[memory:3383754]]，模型导出和推理将优先考虑对Intel NPU/iGPU的优化（如使用OpenVINO）。

### 1.2 系统架构
项目将遵循Ultralytics YOLO框架的标准结构，该结构本身就是模块化和解耦的。

```mermaid
graph TD
    subgraph "数据层 (Data Layer)"
        A[原始图像和标签] --> B{数据预处理与增强};
        B --> C[数据集加载器 (PyTorch DataLoader)];
    end

    subgraph "模型层 (Model Layer)"
        D[YOLOv11 预训练模型 .pt] --> E{模型定义};
        C --> F[训练循环];
        E --> F;
        F --> G[训练好的模型 .pt];
    end

    subgraph "应用层 (Application Layer)"
        G --> H[验证模块 (val.py)];
        G --> I[推理模块 (predict.py)];
        J[新图像/目录] --> I;
        H --> K[性能指标 (mAP, etc.)];
        I --> L[可视化结果/JSON输出];
    end

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
```

### 1.3 数据模型与格式
- **输入数据**:
    - **图像**: 标准图像格式 (`.jpg`, `.png`, `.bmp`)。
    - **标签**: YOLOv5/v8/v11 标准的 `.txt` 格式。每行代表一个边界框，格式为 `<class-index> <x-center-norm> <y-center-norm> <width-norm> <height-norm>`。
- **数据集配置文件 (`.yaml`)**:
    - 这是定义数据集路径、类别数量和类别名称的核心文件。
    - 结构如下:
        ```yaml
        path: ../datasets/tiaozhanbei_datasets_final  # 数据集根目录
        train: images/train  # 训练集图片目录 (相对于path)
        val: images/val    # 验证集图片目录 (相对于path)

        # Classes
        nc: 2  # 类别数量
        names: ['class1', 'class2']  # 类别名称
        ```
- **输出数据**:
    - **模型**: PyTorch的 `.pt` 文件。
    - **推理结果**:
        - 可视化: 在原图上绘制了边界框的图像文件。
        - 结构化: JSON文件，包含每个检测目标的`[x1, y1, x2, y2, confidence, class]`。

## 2. 关键实现策略

### 2.1 数据集分析与选择
- **初步分析**: 首先需要对 `datasets/tiaozhanbei_datasets_1` 和 `datasets/tiaozhanbei_datasets_final` 进行分析。我们将编写一个简短的脚本或手动检查，以确定以下几点：
    - 图像数量和大小分布。
    - 类别定义 (`class.json` 或 `.yaml` 文件)。
    - 标签文件的完整性和正确性。
    - 两个数据集之间的关系（例如，`_final` 是否是 `_1` 的超集或修正版）。
- **选择**: 根据分析结果，我们将**优先选择 `tiaozhanbei_datasets_final`** 作为我们的主要训练和验证数据集，因为它从命名上看更可能是最终版本。我们将在 `TODO.md` 中创建一个任务来确认这一点。

### 2.2 模型选择与训练策略
- **模型选择**: 我们将从 **`yolo11n.pt`** 开始。因为它最小、最快，非常适合用于快速验证整个流程是否通畅。在流程验证成功后，我们会根据对性能（精度）的要求，逐步尝试 `yolo11s.pt` 或 `yolo11m.pt`。
- **训练策略**:
    - **迁移学习**: 所有训练都将基于官方在COCO等大型数据集上预训练好的权重开始，而不是从零开始训练。
    - **超参数**: 初步训练将使用Ultralytics提供的默认超参数。后续会根据验证集表现进行微调。
    - **数据增强**: 将利用YOLOv11内置的强大数据增强功能，如Mosaic、MixUp、CopyPaste等，以提高模型的泛化能力。

### 2.3 脚本实现
- **单一入口**: 为了简化操作，我们将尽量使用`ultralytics`包提供的统一命令行接口（CLI）。而不是为 `train`, `val`, `predict` 创建三个独立的Python脚本。
- **操作示例**:
    - **训练**: `yolo train model=yolo11n.pt data=configs/dataset.yaml epochs=100 imgsz=640`
    - **验证**: `yolo val model=runs/train/exp/weights/best.pt data=configs/dataset.yaml`
    - **推理**: `yolo predict model=runs/train/exp/weights/best.pt source=path/to/image.jpg`
- **配置文件**: 所有特定于本项目的配置（如数据集路径）都将集中在 `configs` 目录下，以保持项目结构的整洁。

## 3. 环境设置

- **依赖安装**: 将提供一个 `requirements.txt` 文件，或者在 `README.md` 中明确指出使用 `pnpm install ultralytics` (或其他等效命令) 来安装所有必要的依赖。
- **CUDA/cuDNN**: `README.md` 将会提醒用户，为了在GPU上获得最佳性能，需要正确安装与PyTorch版本兼容的CUDA和cuDNN。 