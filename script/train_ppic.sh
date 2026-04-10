#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Configuration
GPU_ID=2
DATASET="cifar100"  # Change to "cifar100" for CIFAR-100
BATCH_SIZE=128
EPOCHS=250
LR=0.1
COLOR_WEIGHT="411"

# Privacy budgets 
# EPSILONS=(1.0 2.4 5.2 12.0 20 32.0 58.0)
EPSILONS=(20.0)
echo "================================================"
echo "ResNet-56 Training with Lagrangian DP-Slicing"
echo "================================================"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU_ID}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Color Weight: ${COLOR_WEIGHT}"
echo "Training ${#EPSILONS[@]} privacy budgets"
echo "================================================"

for EPS in "${EPSILONS[@]}"; do
    echo ""
    echo "================================================"
    echo "Training with ε_tot = ${EPS}"
    echo "Started at: $(date)"
    echo "================================================"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m experiment.train_resnet56_ppic \
        --dataset ${DATASET} \
        --epsilon ${EPS} \
        --batchsize ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --milestones 80 160 200 \
        --dp_method "dwt" \
        --wavelet "haar" \
        --color_weight ${COLOR_WEIGHT}

    
    echo "Finished ε_tot = ${EPS} at: $(date)"
    echo ""
done

echo "All training runs completed."