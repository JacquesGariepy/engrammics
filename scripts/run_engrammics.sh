#!/usr/bin/env bash
# =============================================================================
# ONE-CLICK : Distributed Engrammics scientific test.
# Prepares the environment, runs each stage in order without intervention, and
# verifies each result. Stops (exit != 0) if a critical stage fails.
# Everything is logged to engrammics_run.log.
#
#   Usage:                 bash run_engrammics.sh
#   Override model:        MODEL_ID=fla-hub/<name> bash run_engrammics.sh
#   Override LM seed count: SEEDS=50 bash run_engrammics.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # racine du depot (parent de scripts/)
LOG="$HERE/engrammics_run.log"
VENV="$HERE/.venv_engrammics"
PYBIN="${PYTHON:-python3}"
SEEDS="${SEEDS:-30}"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo " DISTRIBUTED ENGRAMMICS -- one-click    $(date -u)"
echo "=============================================================="

# --- 0. OS / Python guards ---------------------------------------------------
case "$(uname -s)" in
  Linux*) : ;;
  *) echo "[!] fla/Triton target Linux + NVIDIA. On Windows, run this under WSL2." ;;
esac
"$PYBIN" --version || { echo "[FATAL] python3 not found"; exit 1; }

# --- 1. Virtual environment --------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "[*] Creating venv: $VENV"
  "$PYBIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

# --- 2. STAGE A: prove the algebra (CPU, no GPU). Critical gate. -------------
echo "[*] Installing numpy"
pip install -q numpy
echo "[*] STAGE A: scientific validation on the toy backend (must report"
echo "    THEORY SUPPORTED and exit 0; this proves the harness and the algebra)"
python "$HERE/src/engrammics_science.py" --backend toy --seeds 60 --quiet
echo "[OK] STAGE A passed: harness validated, algebraic theory supported"

# --- 3. LM dependencies ------------------------------------------------------
echo "[*] Installing LM dependencies (torch pulls the CUDA wheel by default on"
echo "    Linux + NVIDIA; this can take several minutes)"
pip install -q torch
pip install -q flash-linear-attention transformers huggingface_hub safetensors accelerate
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# --- 4. Resolve model --------------------------------------------------------
if [ -z "${MODEL_ID:-}" ]; then
  echo "[*] Discovering a pure DeltaNet checkpoint on fla-hub"
  MODEL_ID="$(python "$HERE/src/pick_model.py" || true)"
fi
if [ -z "${MODEL_ID:-}" ]; then
  echo "[!] No pure DeltaNet found automatically."
  echo "    Re-run with e.g.:  MODEL_ID=fla-hub/<name> bash run_engrammics.sh"
  exit 2
fi
echo "[*] Model: $MODEL_ID"

# --- 5. STAGE B: the real test on the LM ------------------------------------
echo "[*] STAGE B: scientific test on the DeltaNet fast-weight state ($SEEDS seeds)"
set +e
python "$HERE/src/engrammics_science.py" --backend lm --model "$MODEL_ID" --seeds "$SEEDS"
STATUS=$?
set -e

# --- 6. Global verdict -------------------------------------------------------
echo "=============================================================="
case "$STATUS" in
  0) echo " GLOBAL RESULT: THEORY SUPPORTED at LM scale" ;;
  3) echo " GLOBAL RESULT: INCONCLUSIVE (model cannot do the task; not a refutation)" ;;
  *) echo " GLOBAL RESULT: THEORY NOT SUPPORTED on this model/task (see $LOG)" ;;
esac
echo "=============================================================="
exit "$STATUS"
