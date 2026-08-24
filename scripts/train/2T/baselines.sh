#!/bin/bash
set -euo pipefail

# ---- configuration -------------------------------------------------------------
COHORT="2T"
EPOCHS=20
LR=1e-05
DEVICE=${DEVICE:-0}
HEARTBEAT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
OUT=${1:-$HEARTBEAT/logs/train}
# --------------------------------------------------------------------------------

run_fold () {
    ARCH=$1; FOLD=$2

    echo "[$ARCH $COHORT fold $FOLD]  lr=$LR epochs=$EPOCHS"

    CUDA_VISIBLE_DEVICES=$DEVICE OMP_NUM_THREADS=8 "$PYTHON" "$HEARTBEAT/run_train.py" \
        --arch "$ARCH" \
        --cohort "$COHORT" \
        --fold "$FOLD" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --out "$OUT"
}

#         arch       fold
run_fold  vit        1
run_fold  vit        2
run_fold  vit        3
run_fold  vit        4

run_fold  resnet18   1
run_fold  resnet18   2
run_fold  resnet18   3
run_fold  resnet18   4

run_fold  resnet50   1
run_fold  resnet50   2
run_fold  resnet50   3
run_fold  resnet50   4

run_fold  vgg16      1
run_fold  vgg16      2
run_fold  vgg16      3
run_fold  vgg16      4

run_fold  mobilenet  1
run_fold  mobilenet  2
run_fold  mobilenet  3
run_fold  mobilenet  4

echo "done -> $OUT"
