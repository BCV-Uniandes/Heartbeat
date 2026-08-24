import argparse
import json
import os


import torch
from torch.utils.data import DataLoader

from dataloader.dataset import CLASSES
from dataloader.utils import balanced_sampler
from evaluation import write_predictions
from models.build import build
from training import engine
from training.utils import (epoch_logger, iter_progress, load_split, resolve,
                            run_directory, seed_everything, stamp)
from utils import boolean

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train one cross-validation fold and write a checkpoint, a run "
                    "record (JSON) and a plain-text log.")
    ap.add_argument("--arch", default="heart-vit",
                    choices=["heart-vit", "vit", "resnet18", "resnet50", "vgg16",
                             "mobilenet"])
    ap.add_argument("--cohort", required=True, choices=["2T", "3T"])
    ap.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--out",
                    help="run directory: checkpoint, run.json and train.log. Defaults "
                         "logs/train/<cohort>/<arch>/<split>_<fold> beside this file, the "
                         "same shape logs/eval uses. With "
                         "--exp the leaf is named after the hyperparameters instead.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--exp", default=None,
                    help="experiment prefix. With it, --out is treated as a parent "
                         "directory and the run goes in a subdirectory named after the "
                         "prefix and the hyperparameters that distinguish the run")
    ap.add_argument("--dim", type=int, default=None,
                    help="heart-vit only, and required for it: the fold's "
                         "view_embedding_dim")
    ap.add_argument("--adanorm-layers", type=int, default=None,
                    help="heart-vit only, and required for it: the fold's AdaNorm "
                         "block count")
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adanorm-attention-heads", type=int, default=None,
                    help="heart-vit only; defaults to --adanorm-layers, which is what "
                         "every published run used")
    ap.add_argument("--samples-per-class",
                    type=int, nargs=2, metavar=tuple(f"N_{c.upper()}" for c in CLASSES))
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--decoupled-weight-decay",
                    type=boolean, default=True, metavar="true|false")
    ap.add_argument("--keep-resume-state",
                    type=boolean, default=True, metavar="true|false")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="defaults to cuda when available; which GPU is "
                         "CUDA_VISIBLE_DEVICES, as scripts/test/ does it")
    args = ap.parse_args(argv)

    required = ["lr", "epochs"]
    if args.arch == "heart-vit":
        required += ["dim", "adanorm_layers", "dropout"]
    else:
        for name in ("dim", "adanorm_layers", "dropout", "adanorm_attention_heads"):
            if getattr(args, name) is not None:
                ap.error(f"--{name.replace('_', '-')} applies to --arch heart-vit "
                         f"only, not {args.arch}")
    missing = [f"--{n.replace('_', '-')}" for n in required
               if getattr(args, n) is None]
    if missing:
        ap.error(f"--arch {args.arch} requires {', '.join(missing)}")

    out_root = args.out if args.out else os.path.join(HERE, "logs", "train")
    args.out = run_directory(out_root, vars(args), args.exp)

    if args.samples_per_class is not None and min(args.samples_per_class) < 1:
        ap.error(f"--samples-per-class must be two positive counts, got "
                 f"{args.samples_per_class}")

    train, val = load_split(args.dataset, args.cohort, args.fold)
    seed_everything(args.seed)
    sampler = balanced_sampler(train)
    config = resolve(args, train, val, sampler)

    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    device = config["device"]
    model, conditioned = build(
        args.arch, dim=config["view_embedding_dim"],
        layers=config["adanorm_layers"], dropout=config["dropout"],
        heads=config["adanorm_attention_heads"])
    if conditioned != config["conditioned"]:
        raise RuntimeError(
            f"the record says conditioned={config['conditioned']} but "
            f"build returned {conditioned} for --arch {args.arch}")
    model = model.to(device)

    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pth")
    state_path = os.path.join(ckpt_dir, "last.pth")
    record_path = os.path.join(args.out, "run.json")
    predictions_path = os.path.join(args.out, "best_predictions.json")

    if os.path.exists(record_path):
        print(f"{record_path} exists: this run already finished. Delete it to redo the "
              f"run, or pass a different --out.")
        return
    resuming = os.path.exists(state_path)
    if resuming:
        done = torch.load(state_path, map_location="cpu", weights_only=False)["epoch"]
        print(f"resuming {args.out} from epoch {done + 1}/{args.epochs}", flush=True)

    with open(os.path.join(args.out, "train.log"), "a" if resuming else "w") as log:
        log.write(json.dumps(config, indent=1) + "\n")
        log.flush()
        print(json.dumps(config, indent=1), flush=True)

        record = engine.train_fold(
            model, loader, val, epochs=args.epochs, lr=args.lr, device=device,
            conditioned=conditioned, seed=args.seed,
            samples_per_class=config["samples_per_class"], config=config,
            on_epoch=epoch_logger(log, args.epochs),
            on_iter=iter_progress(args.epochs),
            on_val=iter_progress(args.epochs, "val"),
            state_path=state_path, best_path=best_path,
            weight_decay=args.weight_decay,
            decoupled_weight_decay=args.decoupled_weight_decay)

        write_predictions(predictions_path, record.pop("best_rows"))
        with open(record_path, "w") as fh:
            json.dump(record, fh, indent=1)
        if not args.keep_resume_state and os.path.exists(state_path):
            os.remove(state_path)

        best = record["best"]
        summary = (f"{stamp()} best epoch {best['epoch']} (selected on F1): "
                   f"F1 {best['val_f1']:.4f}  thr {best['val_threshold']:.2f}  "
                   f"sens {best['val_sensitivity']:.4f}  "
                   f"spec {best['val_specificity']:.4f}  "
                   f"ppv {best['val_ppv']:.4f}  auroc {best['val_auroc']:.4f}\n"
                   f"{best_path}\n"
                   f"{state_path if args.keep_resume_state else '(resume state dropped)'}"
                   f"\n{predictions_path}\n{record_path}")
        print(summary, flush=True)
        log.write(summary + "\n")


if __name__ == "__main__":
    main()
