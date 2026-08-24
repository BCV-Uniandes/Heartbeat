#!/bin/bash
set -euo pipefail

# ---- configuration -------------------------------------------------------------
ARCH="heart-vit"
COHORT="2T"
EPOCHS=40
DROPOUT=0
DEVICE=${DEVICE:-0}
HEARTBEAT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
OUT=${1:-$HEARTBEAT/logs/train}
# --------------------------------------------------------------------------------

run_fold () {
    FOLD=$1; VIEW_EMB_DIM=$2; LAYERS=$3; HEADS=$4; LR=$5

    echo "[$ARCH $COHORT fold $FOLD]  view_emb_dim=$VIEW_EMB_DIM layers=$LAYERS heads=$HEADS lr=$LR epochs=$EPOCHS"

    CUDA_VISIBLE_DEVICES=$DEVICE OMP_NUM_THREADS=8 "$PYTHON" "$HEARTBEAT/run_train.py" \
        --arch "$ARCH" \
        --cohort "$COHORT" \
        --fold "$FOLD" \
        --dim "$VIEW_EMB_DIM" \
        --adanorm-layers "$LAYERS" \
        --adanorm-attention-heads "$HEADS" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --out "$OUT"
}

#         fold  view_emb_dim  layers  heads  lr
run_fold  1      8            2       2      1e-05
run_fold  2     16            3       3      1e-04
run_fold  3     16            3       3      1e-05
run_fold  4     16            2       2      1e-05

echo "done -> $OUT"
