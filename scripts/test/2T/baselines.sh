#!/bin/bash
set -euo pipefail

# ---- configuration -------------------------------------------------------------
COHORT="2T"
DEVICE=${DEVICE:-0}
HEARTBEAT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python)}
CKPT="$HEARTBEAT/checkpoints/$COHORT"
OUT=${1:-$HEARTBEAT/logs/eval/$COHORT}
# --------------------------------------------------------------------------------

run_fold () {
    ARCH=$1; SPLIT=$2; FOLD=$3; THRESHOLD=${4:-}

    RUN_DIR="$OUT/$ARCH/${SPLIT}/fold_${FOLD}"
    mkdir -p "$RUN_DIR"
    echo "[$ARCH $COHORT $SPLIT fold $FOLD] ${THRESHOLD:+threshold=$THRESHOLD}"

    CUDA_VISIBLE_DEVICES=$DEVICE OMP_NUM_THREADS=8 "$PYTHON" "$HEARTBEAT/run_eval.py" \
        --arch "$ARCH" \
        --cohort "$COHORT" \
        --split "$SPLIT" \
        --fold "$FOLD" \
        --ckpt "$CKPT/$ARCH/${ARCH}_fold_${FOLD}.pth" \
        ${THRESHOLD:+--threshold $THRESHOLD} \
        --out "$RUN_DIR/predictions.json"
}

# Cross-validation sweeps the decision threshold and keeps the best-performing one.
# The test runs then apply that fold's chosen threshold, fixed -- they do not sweep.
#         arch       split      fold  threshold
run_fold  vit        cross_val  1
run_fold  vit        cross_val  2
run_fold  vit        cross_val  3
run_fold  vit        cross_val  4
run_fold  vit        test       1     0.69
run_fold  vit        test       2     0.24
run_fold  vit        test       3     0.51
run_fold  vit        test       4     0.49

run_fold  resnet18   cross_val  1
run_fold  resnet18   cross_val  2
run_fold  resnet18   cross_val  3
run_fold  resnet18   cross_val  4
run_fold  resnet18   test       1     0.80
run_fold  resnet18   test       2     0.64
run_fold  resnet18   test       3     0.69
run_fold  resnet18   test       4     0.46

run_fold  resnet50   cross_val  1
run_fold  resnet50   cross_val  2
run_fold  resnet50   cross_val  3
run_fold  resnet50   cross_val  4
run_fold  resnet50   test       1     0.50
run_fold  resnet50   test       2     0.31
run_fold  resnet50   test       3     0.25
run_fold  resnet50   test       4     0.24

run_fold  vgg16      cross_val  1
run_fold  vgg16      cross_val  2
run_fold  vgg16      cross_val  3
run_fold  vgg16      cross_val  4
run_fold  vgg16      test       1     0.50
run_fold  vgg16      test       2     0.38
run_fold  vgg16      test       3     0.94
run_fold  vgg16      test       4     0.41

run_fold  mobilenet  cross_val  1
run_fold  mobilenet  cross_val  2
run_fold  mobilenet  cross_val  3
run_fold  mobilenet  cross_val  4
run_fold  mobilenet  test       1     0.98
run_fold  mobilenet  test       2     0.94
run_fold  mobilenet  test       3     0.93
run_fold  mobilenet  test       4     0.91

echo "done -> $OUT"
