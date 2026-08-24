import json

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

THRESHOLD_GRID = np.arange(0, 1, 0.01)


def read_predictions(path):
    with open(path) as fh:
        table = json.load(fh)
    codes = list(table)
    return (codes,
            [table[c]["label"] for c in codes],
            [table[c]["prob_chd"] for c in codes])


def f1_at(labels, probs, threshold):
    y_true = np.asarray(labels)
    y_pred = (np.asarray(probs, dtype=float) > threshold).astype(int)
    return float(f1_score(y_true, y_pred, average=None, labels=[0, 1])[1])


def sweep_threshold(labels, probs):
    scores = [f1_at(labels, probs, t) for t in THRESHOLD_GRID]
    best = int(np.argmax(scores))
    return float(THRESHOLD_GRID[best]), float(scores[best])


def evaluate(labels, probs, threshold=None):
    if threshold is None:
        threshold, f1 = sweep_threshold(labels, probs)
    else:
        threshold = float(threshold)
        f1 = f1_at(labels, probs, threshold)

    y_true = np.asarray(labels)
    y_pred = (np.asarray(probs, dtype=float) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "f1": f1,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "auroc": float(roc_auc_score(y_true, np.asarray(probs, dtype=float))),
    }
