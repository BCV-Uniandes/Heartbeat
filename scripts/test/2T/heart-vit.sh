#!/bin/bash
set -euo pipefail

# ---- configuration -------------------------------------------------------------
ARCH="heart-vit"
COHORT="2T"
DEVICE=${DEVICE:-0}                 # GPU index; several scripts can run at once on different GPUs
HEARTBEAT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
CKPT="$HEARTBEAT/checkpoints/$COHORT/$ARCH"
OUT=${1:-$HEARTBEAT/logs/eval/$COHORT/$ARCH}
# --------------------------------------------------------------------------------

run_fold () {
    SPLIT=$1; FOLD=$2; VIEW_EMB_DIM=$3; LAYERS=$4; HEADS=$5; THRESHOLD=${6:-}

    RUN_DIR="$OUT/${SPLIT}/fold_${FOLD}"
    mkdir -p "$RUN_DIR"
    echo "[$ARCH $COHORT $SPLIT fold $FOLD]  view_emb_dim=$VIEW_EMB_DIM layers=$LAYERS heads=$HEADS ${THRESHOLD:+threshold=$THRESHOLD}"

    CUDA_VISIBLE_DEVICES=$DEVICE OMP_NUM_THREADS=8 "$PYTHON" "$HEARTBEAT/run_eval.py" \
        --arch "$ARCH" \
        --cohort "$COHORT" \
        --split "$SPLIT" \
        --fold "$FOLD" \
        --dim "$VIEW_EMB_DIM" \
        --adanorm-layers "$LAYERS" \
        --adanorm-attention-heads "$HEADS" \
        --ckpt "$CKPT/${ARCH}_fold_${FOLD}.pth" \
        ${THRESHOLD:+--threshold $THRESHOLD} \
        --out "$RUN_DIR/predictions.json"
}

# Cross-validation sweeps the decision threshold and keeps the best-performing one.
# The test runs then apply that fold's chosen threshold, fixed -- they do not sweep.
#         split      fold  view_emb_dim  layers  heads  threshold
run_fold  cross_val  1      8            2       2
run_fold  cross_val  2     16            3       3
run_fold  cross_val  3     16            3       3
run_fold  cross_val  4     16            2       2

run_fold  test       1      8            2       2      0.44
run_fold  test       2     16            3       3      0.03
run_fold  test       3     16            3       3      0.47
run_fold  test       4     16            2       2      0.33

echo "done -> $OUT"
