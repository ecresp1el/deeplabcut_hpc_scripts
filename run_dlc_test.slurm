#!/bin/bash
#SBATCH --job-name=DeepLabCut_test
#SBATCH --account=parent0
#SBATCH --partition=gpu           # GPU partition on Great Lakes
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=elcrespo@umich.edu
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "===== [$(date)] Starting DeepLabCut GPU test on Great Lakes ====="

########################
# USER-CONFIGURABLE VARS
########################
PROJECT_BASENAME="DeepLabCut_project"       # name stem for the Turbo subfolder
ENV_NAME="deeplabcut-env"                   # conda env name (recreated each run)
PYTHON_VERSION="3.10"                       # DLC 3.x supports Python >=3.10
TORCH_CUDA="11.8"                           # CUDA runtime for PyTorch (adjust if needed)

# Storage locations
shared_turbo_directory="/nfs/turbo/umms-parent"
timestamp=$(date +"%Y%m%d-%H%M%S")
dlc_project_path="${shared_turbo_directory}/${PROJECT_BASENAME}-${timestamp}"

echo "Shared Turbo root : ${shared_turbo_directory}"
echo "Run project folder: ${dlc_project_path}"
echo "NOTE: Only creating/using the subfolder above; nothing else in /nfs/turbo/umms-parent will be modified."

echo "===== [$(date)] Creating project directory on Turbo ====="
mkdir -p "${dlc_project_path}"
cd "${dlc_project_path}"

########################
# MODULES + VENV SETUP
########################
echo "===== [$(date)] Loading Anaconda module ====="
module purge
module load python3.11-anaconda/2024.02
echo "✅ Loaded Anaconda module"
echo "===== [$(date)] Loading PyTorch module ====="
module load pytorch/2.0.1
echo "✅ Loaded PyTorch module"
echo "===== [$(date)] Currently loaded modules ====="
module list

########################
# CREATE PYTHON VENV
########################
echo "===== [$(date)] Creating Python venv at ${dlc_project_path}/dlc-venv ====="
python -m venv dlc-venv
echo "✅ Created Python venv"

echo "===== [$(date)] Activating Python venv ====="
source dlc-venv/bin/activate
echo "✅ Activated Python venv"

echo "===== [$(date)] Upgrading pip ====="
pip install --upgrade pip
echo "✅ Upgraded pip"

echo "===== [$(date)] Installing DeepLabCut (pip, headless) ====="
pip install --pre "deeplabcut"
echo "✅ DLC installed"

echo "===== [$(date)] Python & package versions ====="
python -V
python - <<'PY'
import sys, subprocess
print("Python:", sys.version)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device 0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check error:", e)
try:
    import deeplabcut
    print("DeepLabCut:", deeplabcut.__version__)
except Exception as e:
    print("DLC import error:", e)
PY

########################
# GET DLC TEST & RUN IT
########################
echo "===== [$(date)] Cloning DeepLabCut repository (for testscript_cli.py) ====="
git clone https://github.com/DeepLabCut/DeepLabCut.git
cd DeepLabCut

echo "===== [$(date)] Running DeepLabCut testscript_cli.py ====="
# This script creates a small test project, extracts frames, and runs a short training check.
python testscript_cli.py

echo "===== [$(date)] Test finished. Listing contents for inspection ====="
ls -lah
echo "Output files are under: ${dlc_project_path}"

echo "===== [$(date)] Done. Check slurm-${SLURM_JOB_ID}.out for full logs. ====="