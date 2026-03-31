#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Dataset: paths must match FileListDataset in experiment/train_arcface_ppfr.py
#   --data_root   root folder for images
#   --file_list   text file with lines: <relative_path> <class_id>
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-/path/to/ms1m-arcface}"
FILE_LIST="${FILE_LIST:-${DATA_ROOT}/ms1m_arcface_file_list.txt}"

# IR-50 pretrained backbone 
PRETRAINED_PATH="${PRETRAINED_PATH:-}"

# Parallel training
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WORLD_SIZE="${WORLD_SIZE:-4}"

EPOCHS=24
BATCH_SIZE=384
COLOR_WEIGHT="411"
ABLATION="lagrangian"

DP_METHOD="dwt"
WAVELET="haar"

EPSILONS=(20.0)

echo "================================================"
echo "ArcFace (PPFR) training with DP-Slicing"
echo "================================================"
echo "Repo: ${REPO_ROOT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "FILE_LIST: ${FILE_LIST}"
echo "PRETRAINED_PATH: ${PRETRAINED_PATH:-<none>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "DP_METHOD: ${DP_METHOD}"
if [[ "${DP_METHOD}" == "dwt" ]]; then
  echo "WAVELET: ${WAVELET}"
fi
echo "COLOR_WEIGHT: ${COLOR_WEIGHT} (411/211/111)"
echo "ABLATION: ${ABLATION}"
echo "EPOCHS: ${EPOCHS}  BATCH_SIZE (global): ${BATCH_SIZE}"
echo "Training ${#EPSILONS[@]} privacy budget(s)"
echo "================================================"

for EPS in "${EPSILONS[@]}"; do
  echo ""
  echo "================================================"
  echo "Training with ε_tot = ${EPS}"
  echo "Started at: $(date)"
  echo "================================================"

  CMD=(
    python -m experiment.train_arcface_ppfr
    --data_root "${DATA_ROOT}"
    --file_list "${FILE_LIST}"
    --dp_method "${DP_METHOD}"
    --wavelet "${WAVELET}"
    --epsilon "${EPS}"
    --color_weight "${COLOR_WEIGHT}"
    --ablation "${ABLATION}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr 0.01
    --warmup_epochs 0
    --warmup_start_lr 1e-5
    --lr_milestones 10 18 22
    --lr_gamma 0.1
    --arcface_s 64.0
    --arcface_m 0.5
    --num_workers 8
    --world_size "${WORLD_SIZE}"
  )

  if [[ -n "${PRETRAINED_PATH}" ]]; then
    CMD+=(--pretrained_path "${PRETRAINED_PATH}")
  fi

  "${CMD[@]}"

  echo "Finished ε_tot = ${EPS} at: $(date)"
  echo ""
done

echo "All PPFR training runs completed."
