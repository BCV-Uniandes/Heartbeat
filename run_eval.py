import argparse
import csv
import os
import traceback

import torch

import metrics
from dataloader import DelfosPatientDataset
from evaluation import logger, patient_progress, predict, write_predictions
from models.build import build
from utils import device_name

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="heart-vit",
                    choices=["heart-vit", "vit", "resnet18", "resnet50", "vgg16",
                             "mobilenet"])
    ap.add_argument("--cohort", required=True, choices=["2T", "3T"])
    ap.add_argument("--split", required=True, choices=["cross_val", "test"])
    ap.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="predictions JSON to write")
    ap.add_argument("--log", default=None,
                    help="log file; defaults to test.log beside --out")
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--dim", type=int, default=None,
                    help="heart-vit only, and required for it: the fold's "
                         "view_embedding_dim, as scripts/train/heart-vit_*.sh passes it")
    ap.add_argument("--adanorm-layers", type=int, default=None,
                    help="heart-vit only, and required for it: the fold's AdaNorm "
                         "block count")
    ap.add_argument("--adanorm-attention-heads", type=int, default=None,
                    help="heart-vit only; defaults to --adanorm-layers, which is what "
                         "every published run used")
    args = ap.parse_args()

    if args.cohort == "3T" and args.split == "test":
        ap.error("no test split exists for cohort 3T (only 2T has dataset/2T/test)")

    if args.arch == "heart-vit":
        missing = [f"--{n.replace('_', '-')}" for n in ("dim", "adanorm_layers")
                   if getattr(args, n) is None]
        if missing:
            ap.error(f"--arch heart-vit requires {' and '.join(missing)}")
    else:
        for name in ("dim", "adanorm_layers", "adanorm_attention_heads"):
            if getattr(args, name) is not None:
                ap.error(f"--{name.replace('_', '-')} applies to --arch heart-vit "
                         f"only, not {args.arch}")

    if not os.path.isdir(args.dataset):
        ap.error(f"dataset not found at {args.dataset}\n")
    if not os.path.exists(args.ckpt):
        ap.error(f"checkpoint not found at {args.ckpt}\n")

    root = os.path.join(args.dataset, args.cohort, args.split)
    codes = None
    if args.split == "cross_val":
        with open(os.path.join(root, f"fold{args.fold}.csv")) as fh:
            codes = {r["patient_code"] for r in csv.DictReader(fh) if r["split"] == "val"}

    data = DelfosPatientDataset(root, os.path.join(root, "metadata_normalized.csv"),
                                patient_codes=codes)

    model, conditioned = build(args.arch, dim=args.dim, layers=args.adanorm_layers,
                               heads=args.adanorm_attention_heads)
    device = device_name()
    model = model.to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    tag = f"[{args.arch} {args.cohort} {args.split} {args.fold}]"
    log_path = args.log or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                        "test.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        emit = logger(log, tag)
        try:
            emit(f"{len(data)} patients from {root}")
            rows = predict(model, data, device=device, conditioned=conditioned,
                           on_patient=patient_progress(emit))
            write_predictions(args.out, rows)

            labels = [label for _code, label, _prob, _per_image in rows]
            probs = [prob for _code, _label, prob, _per_image in rows]
            scored = metrics.evaluate(labels, probs, threshold=args.threshold)
            emit(f"{args.out}: {len(rows)} patients")
            emit(f"threshold {scored['threshold']:.4f} "
                 f"({'fixed' if args.threshold is not None else 'swept'})")
            emit("  ".join(f"{key} {scored[key]:.4f}" for key in
                           ("f1", "sensitivity", "specificity", "ppv", "auroc")))
        except BaseException:
            log.write(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
