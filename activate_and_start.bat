@echo off
REM ==============================================================================
REM YOLOv11 训练一键启动器 (v2.0 - 集成.sh逻辑)
REM ==============================================================================
REM
REM 功能:
REM 1. 自动激活指定的 Conda 环境。
REM 2. [集成逻辑] 检查数据集、光流、配置文件，如果不存在则自动执行预处理脚本。
REM 3. [集成逻辑] 检查预训练权重文件是否存在。
REM 4. [集成逻辑] 使用所有必需的参数调用 train_yolov11.py 启动训练。
REM
REM 使用方法:
REM   直接双击运行，或在 cmd/powershell 中执行此文件。
REM   可以传递额外参数，例如: activate_and_start.bat --epochs 150 --batch 8
REM
REM ==============================================================================

REM --- 配置区 ---
REM 请根据您的实际情况，修改以下 Miniconda/Anaconda 的安装路径和环境名称
SET CONDA_INSTALL_PATH=D:\miniconda3
SET CONDA_ENV_NAME=yolo_env

REM --- 激活环境 ---
ECHO [INFO] 正在准备启动环境...
call "%CONDA_INSTALL_PATH%\Scripts\activate.bat" %CONDA_ENV_NAME%
if %errorlevel% neq 0 (
    ECHO [ERROR] Conda 环境激活失败!
    ECHO [ERROR] 请检查 CONDA_INSTALL_PATH 和 CONDA_ENV_NAME 设置是否正确。
    pause
    exit /b %errorlevel%
)
ECHO [SUCCESS] Conda 环境激活成功!

REM --- 路径与配置定义 (从 start_training.sh 翻译而来) ---
REM %~dp0 会扩展为包含此 .bat 文件的驱动器号和路径 (e.g., D:\Yolo\Yolo-for-GitHub\)
SET "SCRIPT_DIR=%~dp0"
REM 获取上级目录 (e.g., D:\Yolo\)
for %%F in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fF"

SET "PREPROCESSING_SCRIPTS_DIR=%PROJECT_ROOT%\scripts\data_preprocessing"

REM 最终需要检查的关键文件和目录
SET "FINAL_DATASET_DIR=%PROJECT_ROOT%\datasets\balanced_tiaozhanbei_split"
SET "FINAL_FLOW_DIR=%FINAL_DATASET_DIR%\flow"
SET "FINAL_CONFIG_FILE=%SCRIPT_DIR%configs\tiaozhanbei_final.yaml"

REM 模型与训练参数
SET "PRETRAINED_WEIGHTS=%SCRIPT_DIR%models\yolo11x.pt"
SET "MODEL_CONFIG=%SCRIPT_DIR%configs\yolo11-5ch.yaml"
SET "DATA_CONFIG=%FINAL_CONFIG_FILE%"
SET "EPOCHS=100"
SET "BATCH_SIZE=8"
SET "IMG_SIZE=640"
SET "PROJECT_NAME=runs\train"
SET "EXPERIMENT_NAME=yolo11x_5channel_exp1"
SET "DEVICE=0"

ECHO [INFO] 已设置所有路径和参数。

REM --- 自动化预处理检查与执行 ---
ECHO.
ECHO [1/4] 正在检查数据集和配置文件...

SET "PREPROCESSING_NEEDED=0"
if not exist "%FINAL_FLOW_DIR%\" ( SET "PREPROCESSING_NEEDED=1" )
if not exist "%FINAL_CONFIG_FILE%" ( SET "PREPROCESSING_NEEDED=1" )

if %PREPROCESSING_NEEDED% == 1 (
    ECHO [WARNING] 检测到部分或全部预处理产物缺失，将启动全自动预处理流程...

    ECHO   [Step A] 按清单整理文件...
    python "%PREPROCESSING_SCRIPTS_DIR%\organize_files.py"
    if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
    ECHO   [SUCCESS] 文件整理完成。

    ECHO   [Step B] 批量计算光流 (这可能需要较长时间)...
    python "%PREPROCESSING_SCRIPTS_DIR%\compute_optical_flow.py"
    if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
    ECHO   [SUCCESS] 光流计算完成。

    ECHO   [Step C] 创建最终配置文件...
    python "%PREPROCESSING_SCRIPTS_DIR%\create_final_yaml.py"
    if %errorlevel% neq 0 ( pause & exit /b %errorlevel% )
    ECHO   [SUCCESS] 配置文件创建完成。

    ECHO [SUCCESS] 全自动预处理流程执行完毕！
) ELSE (
    ECHO [SUCCESS] 数据集和配置文件均已就绪。
)

REM --- 检查预训练权重文件 ---
ECHO.
ECHO [2/4] 正在检查预训练权重...
if not exist "%PRETRAINED_WEIGHTS%" (
    ECHO [ERROR] 预训练权重文件未找到！
    ECHO   期望路径: %PRETRAINED_WEIGHTS%
    ECHO   请将 yolo11x.pt 模型文件放置在 Yolo-for-GitHub\models\ 目录下。
    pause
    exit /b 1
)
ECHO [SUCCESS] 预训练权重已找到: %PRETRAINED_WEIGHTS%

REM --- 启动训练 ---
ECHO.
ECHO [3/4] 准备启动训练...

ECHO ==============================================================================
ECHO   YOLOv11 5-Channel Training Starting
ECHO ==============================================================================
ECHO ^> Model Config:    %MODEL_CONFIG%
ECHO ^> Pretrained:      %PRETRAINED_WEIGHTS%
ECHO ^> Dataset Config:  %DATA_CONFIG%
ECHO ^> Epochs:          %EPOCHS%
ECHO ^> Batch Size:      %BATCH_SIZE%
ECHO ^> Image Size:      %IMG_SIZE%
ECHO ^> Project:         %PROJECT_NAME%
ECHO ^> Experiment:      %EXPERIMENT_NAME%
ECHO ^> Device:          %DEVICE%
ECHO ^> Extra Args:      %*
ECHO ------------------------------------------------------------------------------

REM 切换到脚本所在目录执行，确保相对路径正确
cd /d "%SCRIPT_DIR%"
ECHO [INFO] 已切换到工作目录: %cd%

ECHO.
ECHO [4/4] 执行训练命令...
python train_yolov11.py ^
    --model "%MODEL_CONFIG%" ^
    --weights "%PRETRAINED_WEIGHTS%" ^
    --data "%DATA_CONFIG%" ^
    --epochs "%EPOCHS%" ^
    --batch "%BATCH_SIZE%" ^
    --imgsz "%IMG_SIZE%" ^
    --project "%PROJECT_NAME%" ^
    --name "%EXPERIMENT_NAME%" ^
    --device "%DEVICE%" ^
    %*

ECHO ==============================================================================
ECHO [SUCCESS] 训练脚本已执行完毕。

REM 保持窗口打开，以便查看日志
pause 