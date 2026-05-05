"""
垃圾分类项目 - 创建 conda 环境 gc_gpu
=====================================================================
运行方式：
    python create_env.py

注意：直接用系统 Python 或任意 conda base 环境运行均可，
      不需要提前激活 gc_gpu。
=====================================================================
"""

import subprocess
import sys
import os
import re
import platform


# ── 要安装的依赖 ──────────────────────────────────────────────────
ENV_NAME    = "gc_gpu"
PYTHON_VER  = "3.10"

DEPS = [
    "numpy==1.26.4",
    "scikit-learn==1.4.2",
    "Pillow==10.3.0",
    "matplotlib==3.9.0",
    "tqdm==4.66.4",
    "opencv-python==4.9.0.80",
]

TF_CUDA12 = "tensorflow[and-cuda]==2.15.1"
TF_CUDA11 = "tensorflow==2.13.1"


# ── 工具函数 ──────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """运行命令并实时显示输出"""
    print(f"\n  >>> {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"\n  [警告] 命令退出码 {result.returncode}，请查看上方输出")
    return result


def run_in_env(pip_cmd: str) -> bool:
    """在 gc_gpu 环境中运行 pip 命令"""
    full = f'conda run -n {ENV_NAME} --no-capture-output python -m {pip_cmd}'
    return run(full, check=False).returncode == 0


# ── 检测函数 ──────────────────────────────────────────────────────
def check_conda() -> str:
    """返回 conda 可执行文件路径，找不到则退出"""
    section("检测 conda")
    ret = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
    if ret.returncode != 0:
        print("  [错误] 未找到 conda 命令！")
        print("  请先安装 Anaconda 或 Miniconda：")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        input("\n  按回车键退出...")
        sys.exit(1)
    print(f"  ✓ {ret.stdout.strip()}")
    return "conda"


def check_nvidia_gpu() -> bool:
    section("检测 NVIDIA GPU")
    ret = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if ret.returncode == 0:
        for line in ret.stdout.splitlines()[:12]:
            print(f"  {line}")
        return True
    print("  未检测到 NVIDIA GPU（nvidia-smi 不可用）")
    print("  将安装 CPU 版 TensorFlow")
    return False


def get_cuda_major(has_gpu: bool):
    if not has_gpu:
        return None
    ret = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    match = re.search(r"CUDA Version:\s*(\d+)\.", ret.stdout)
    if match:
        major = int(match.group(1))
        print(f"\n  检测到 CUDA 主版本：{major}.x")
        return major
    return None


def pick_tf(has_gpu: bool, cuda_major) -> str:
    section("选择 TensorFlow GPU 版本")
    if not has_gpu or cuda_major is None:
        # 未检测到 GPU 时仍安装 GPU 版（tensorflow 本身支持 GPU，
        # 只要后续正确安装 CUDA 驱动即可自动启用）
        print("  ⚠  未检测到 GPU，但仍安装 GPU 版 TensorFlow")
        print("     确保已安装 NVIDIA 驱动 + CUDA 11.8 后即可启用 GPU 加速")
        print(f"  → {TF_CUDA11}")
        return TF_CUDA11
    if cuda_major >= 12:
        print(f"  → CUDA {cuda_major}.x 检测到，安装：{TF_CUDA12}")
        print("    (tensorflow[and-cuda] 会自动安装匹配的 cudatoolkit / cudnn)")
        return TF_CUDA12
    print(f"  → CUDA {cuda_major}.x 检测到，安装：{TF_CUDA11}")
    print("    (内置 GPU 支持，需系统已安装 CUDA 11.8 + cuDNN 8.6)")
    return TF_CUDA11


# ── 核心流程 ──────────────────────────────────────────────────────
def create_env():
    section(f"创建 conda 环境：{ENV_NAME}（Python {PYTHON_VER}）")

    # 若环境已存在，询问是否重建
    ret = subprocess.run(
        f"conda env list", shell=True, capture_output=True, text=True
    )
    if ENV_NAME in ret.stdout:
        print(f"  环境 [{ENV_NAME}] 已存在")
        ans = input("  是否删除并重建？[y/N] ").strip().lower()
        if ans == "y":
            run(f"conda env remove -n {ENV_NAME} -y")
        else:
            print("  跳过创建，继续安装/更新依赖...")
            return

    run(f"conda create -n {ENV_NAME} python={PYTHON_VER} -y", check=True)
    print(f"\n  ✓ 环境 [{ENV_NAME}] 创建成功")


def install_deps(tf_pkg: str):
    section("升级 pip")
    run_in_env("pip install --upgrade pip setuptools wheel")

    section(f"安装 TensorFlow：{tf_pkg}")
    run_in_env(f'pip install "{tf_pkg}"')

    section("安装训练依赖")
    for dep in DEPS:
        run_in_env(f'pip install "{dep}"')


def verify():
    section("验证安装结果")
    script = (
        "import tensorflow as tf, sklearn, PIL, matplotlib, numpy as np; "
        "print('  TensorFlow :', tf.__version__); "
        "print('  NumPy      :', np.__version__); "
        "print('  sklearn    :', sklearn.__version__); "
        "print('  Pillow     :', PIL.__version__); "
        "print('  Matplotlib :', matplotlib.__version__); "
        "gpus = tf.config.list_physical_devices('GPU'); "
        "print('  GPU 数量   :', len(gpus), '✓ GPU加速可用' if gpus else '（CPU模式）'); "
        "[print(f'    {g}') for g in gpus]"
    )
    run(f'conda run -n {ENV_NAME} --no-capture-output python -c "{script}"', check=False)


def print_next():
    section("全部完成！")
    project = os.path.dirname(os.path.abspath(__file__))
    print(f"""
  以后每次训练前，先在终端激活环境：

      conda activate {ENV_NAME}

  然后进入 train 目录执行：

      cd "{project}\\train"
      python train_domestic.py       ← 训练国内四分类
      python train_international.py  ← 训练国际十分类
      python convert_tflite.py       ← 转 TFLite 并自动复制到 Android

  如只想快速测试 App 界面（无需数据集）：

      cd "{project}\\train"
      python create_placeholder_models.py
""")


# ── 入口 ──────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 60)
    print("  垃圾分类识别项目 ─ 创建 conda 环境 gc_gpu")
    print(f"  系统：{platform.system()} {platform.machine()}")
    print(f"  当前 Python：{sys.version.split()[0]}")
    print("=" * 60)

    check_conda()
    has_gpu    = check_nvidia_gpu()
    cuda_major = get_cuda_major(has_gpu)
    tf_pkg     = pick_tf(has_gpu, cuda_major)

    create_env()
    install_deps(tf_pkg)
    verify()
    print_next()

    input("\n  按回车键退出...")


if __name__ == "__main__":
    main()
