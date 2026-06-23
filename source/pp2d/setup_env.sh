#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pp2d"
PYTHON_VERSION="3.12"
INSTALL_TORCH_STACK="${PP2D_INSTALL_TORCH_STACK:-0}"
SKIP_CREATE="${PP2D_SKIP_CREATE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# Activate env (shell must be initialized for conda)
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ "$SKIP_CREATE" != "1" ]]; then
  # Create env
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

# Core deps from requirements.txt
python -m pip install --upgrade pip
if [[ "$INSTALL_TORCH_STACK" == "1" ]]; then
  python -m pip install -r requirements.txt
else
  if ! python - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import torch_geometric  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
torch and torch_geometric are not installed in this environment.

Install the CUDA-matched torch stack first, then rerun:

  conda activate pp2d
  # install torch / torch_geometric for your CUDA version
  PP2D_SKIP_CREATE=1 bash setup_env.sh

Alternatively set PP2D_INSTALL_TORCH_STACK=1 to install the unpinned torch stack
recorded in requirements.txt.
EOF
    exit 2
  fi
  tmp_requirements="$(mktemp)"
  trap 'rm -f "$tmp_requirements"' EXIT
  grep -Ev '^(torch|torch_geometric)([[:space:]]|$|[<>=~!])' requirements.txt > "$tmp_requirements"
  python -m pip install -r "$tmp_requirements"
fi

# Project-specific conda deps (docking & utils)
conda install -y -c conda-forge rdkit openbabel tensorboard pyyaml easydict python-lmdb

# Vina docking and helpers
python -m pip install meeko scipy pdb2pqr vina
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# Optional: if you use these scripts, install extras
python -m pip install biopython tqdm requests nltk rouge-score gemmi

# NOTE:
# - torch and torch_geometric wheels depend on your CUDA driver/runtime.
#   By default this helper skips them even though they are recorded in
#   requirements.txt. Install the torch stack after choosing the CUDA build, or
#   rerun with PP2D_INSTALL_TORCH_STACK=1 if requirements.txt already matches
#   the target machine.
