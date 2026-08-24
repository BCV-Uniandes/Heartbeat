#!/bin/bash
set -euo pipefail

# ---- configuration -------------------------------------------------------------
ARCH="heart-vit"
COHORT="3T"
DEVICE=${DEVICE:-0}
HEARTBEAT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
CKPT="$HEARTBEAT/checkpoints/$COHORT/$ARCH"
OUT=${1:-$HEARTBEAT/logs/eval/$COHORT/$ARCH}
# --------------------------------------------------------------------------------

run_fold () {
    SPLIT=$1; FOLD=$2; VIEW_EMB_DIM=$3; LAYERS=$4; HEADS=$5

    RUN_DIR="$OUT/${SPLIT}/fold_${FOLD}"
    mkdir -p "$RUN_DIR"
    echo "[$ARCH $COHORT $SPLIT fold $FOLD]  view_emb_dim=$VIEW_EMB_DIM layers=$LAYERS heads=$HEADS"

    CUDA_VISIBLE_DEVICES=$DEVICE OMP_NUM_THREADS=8 "$PYTHON" "$HEARTBEAT/run_eval.py" \
        --arch "$ARCH" \
        --cohort "$COHORT" \
        --split "$SPLIT" \
        --fold "$FOLD" \
        --dim "$VIEW_EMB_DIM" \
        --adanorm-layers "$LAYERS" \
        --adanorm-attention-heads "$HEADS" \
        --ckpt "$CKPT/${ARCH}_fold_${FOLD}.pth" \
        --out "$RUN_DIR/predictions.json"
}

# Cross-validation sweeps the decision threshold and keeps the best-performing one.
#         split      fold  view_emb_dim  layers  heads
run_fold  cross_val  1     16            2       2
run_fold  cross_val  2     16            3       3
run_fold  cross_val  3     16            3       3
run_fold  cross_val  4     16            3       3

echo "done -> $OUT"
