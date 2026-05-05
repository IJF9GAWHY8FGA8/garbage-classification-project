@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   垃圾分类项目 - 创建 conda 环境 gc_gpu
echo ============================================================
echo.

:: 检查 conda 是否可用
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 conda 命令，请先安装 Anaconda 或 Miniconda
    echo 下载地址：https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/6] 检测 conda 版本...
conda --version

echo.
echo [2/6] 创建环境 gc_gpu（Python 3.10）...
conda create -n gc_gpu python=3.10 -y
if %errorlevel% neq 0 (
    echo [错误] 环境创建失败，请检查 conda 是否正常
    pause
    exit /b 1
)

echo.
echo [3/6] 激活环境 gc_gpu...
call conda activate gc_gpu
if %errorlevel% neq 0 (
    echo [错误] 环境激活失败
    pause
    exit /b 1
)

echo.
echo [4/6] 升级 pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [5/6] 检测 GPU 并安装 TensorFlow...

:: 检测 CUDA 版本
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 未检测到 NVIDIA GPU，安装 CPU 版 TensorFlow
    pip install tensorflow-cpu==2.13.1
) else (
    :: 获取 CUDA 主版本
    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%i
    echo [提示] 检测到 CUDA 版本：%CUDA_VER%

    echo %CUDA_VER% | findstr /b "12" >nul
    if %errorlevel% equ 0 (
        echo [提示] CUDA 12.x - 安装 tensorflow[and-cuda]==2.15.1
        pip install "tensorflow[and-cuda]==2.15.1"
    ) else (
        echo [提示] CUDA 11.x - 安装 tensorflow==2.13.1
        pip install tensorflow==2.13.1
    )
)

echo.
echo [6/6] 安装训练依赖...
pip install numpy==1.26.4
pip install scikit-learn==1.4.2
pip install Pillow==10.3.0
pip install matplotlib==3.9.0
pip install tqdm==4.66.4
pip install opencv-python==4.9.0.80

echo.
echo ============================================================
echo   验证安装结果
echo ============================================================
python -c "import tensorflow as tf; import sklearn, PIL, matplotlib, numpy as np; print('TensorFlow :', tf.__version__); print('NumPy      :', np.__version__); print('sklearn    :', sklearn.__version__); print('Pillow     :', PIL.__version__); gpus=tf.config.list_physical_devices('GPU'); print('GPU 数量   :', len(gpus), '( GPU加速可用 )' if gpus else '( CPU模式 )')"

echo.
echo ============================================================
echo   环境创建完成！
echo ============================================================
echo.
echo 以后每次使用前，先在终端运行：
echo     conda activate gc_gpu
echo.
echo 然后进入 train 目录开始训练：
echo     cd "c:\Users\14681\Desktop\BS\garbage classification\train"
echo     python train_domestic.py
echo     python train_international.py
echo     python convert_tflite.py
echo.
echo ============================================================
pause
