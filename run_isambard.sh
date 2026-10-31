#!/bin/bash
#SBATCH --job-name=fvm_solver
#SBATCH --output=fvm_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Submit from the repo root. Arguments after the script name go to run_isambard.py:
#   sbatch run_isambard.sh --n-iter 500
#   sbatch --time=12:00:00 run_isambard.sh --save-t 0.1
#   sbatch --partition=hopper run_isambard.sh --n-iter 500    # x86 H100 node on MACS

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export MPLBACKEND=Agg

# Separate environments because Isambard-AI is aarch64 and MACS is x86_64.
VENV=".venv/$(uname -m)"
PY="$VENV/bin/python"

if [ ! -f "$VENV/.installed" ]; then
    uv venv --seed --python=3.12 "$VENV"
    uv pip install --python "$PY" "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
    grep -vE '^torch' requirements.txt > "$VENV/requirements-rest.txt"
    uv pip install --python "$PY" -r "$VENV/requirements-rest.txt"
    touch "$VENV/.installed"
fi

nvidia-smi --list-gpus
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

"$PY" run_isambard.py "$@"
