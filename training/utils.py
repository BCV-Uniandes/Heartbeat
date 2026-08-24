import csv
import datetime
import os
import random
import sys

import numpy as np
import torch

from dataloader import DelfosImageDataset, DelfosPatientDataset
from dataloader.dataset import CLASSES
from dataloader.utils import class_counts
from utils import device_name

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_split(dataset_root, cohort, fold):
    root = os.path.join(dataset_root, cohort, "cross_val")
    metadata = os.path.join(root, "metadata_normalized.csv")
    fold_csv = os.path.join(root, f"fold{fold}.csv")
    if not os.path.isfile(fold_csv):
        raise FileNotFoundError(
            f"{fold_csv} does not exist; --dataset must point at a tree shaped like "
            "this repo's Heartbeat/dataset/ (cohort/cross_val/fold{1..4}.csv)")
    train = DelfosImageDataset(root, metadata, fold_csv, split="train")
    with open(fold_csv) as fh:
        codes = {r["patient_code"] for r in csv.DictReader(fh) if r["split"] == "val"}
    val = DelfosPatientDataset(root, metadata, patient_codes=codes)
    return train, val

def geometry(args):
    if args.arch != "heart-vit":
        return {"view_embedding_dim": None, "adanorm_layers": None, "dropout": None,
                "adanorm_attention_heads": None}
    return {"view_embedding_dim": args.dim,
            "adanorm_layers": args.adanorm_layers,
            "dropout": args.dropout,
            "adanorm_attention_heads": (args.adanorm_layers
                                        if args.adanorm_attention_heads is None
                                        else args.adanorm_attention_heads)}
def run_dirname(config, prefix=None):
    if not prefix:
        return ""
    parts = [prefix]
    if config.get("dropout") is not None:
        parts.append(f"do{config['dropout']}")
    parts += [f"lr{config['lr']:g}", f"ep{config['epochs']}", f"seed{config['seed']}"]
    if config.get("weight_decay"):
        parts.append(f"wd{config['weight_decay']:g}")
        if not config.get("decoupled_weight_decay", True):
            parts.append("coupledwd")
    return "_".join(parts)


def run_directory(root, config, prefix=None):
    path = os.path.join(root, config["cohort"], config["arch"],
                        config.get("split", "cross_val"), f"fold_{config['fold']}")
    leaf = run_dirname(config, prefix)
    return os.path.join(path, leaf) if leaf else path
def resolve(args, train, val, sampler):
    if args.samples_per_class is not None:
        counts, source = list(args.samples_per_class), "--samples-per-class"
    else:
        counts, source = class_counts(train), "counted from the training split"
    config = {
        "arch": args.arch,
        "cohort": args.cohort,
        "fold": args.fold,
        "split": "cross_val",
        "dataset": os.path.abspath(args.dataset),
        "out": os.path.abspath(args.out),
        "pretrained": True,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "samples_per_class": counts,
        "samples_per_class_source": source,
        "train_images": len(train),
        "val_patients": len(val),
        "sampler": type(sampler).__name__,
        "samples_per_epoch": len(sampler),
        "train_transform": train.transform_name,
        "conditioned": args.arch == "heart-vit",
        "weight_decay": args.weight_decay,
        "decoupled_weight_decay": args.decoupled_weight_decay,
        "device": device_name(args.device),
    }
    config.update(geometry(args))
    return config
def stamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
def iter_progress(epochs, phase="iter", stream=None):

    stream = stream if stream is not None else sys.stdout
    tty = stream.isatty()

    def on_iter(epoch, index, total, loss=None):
        text = f"epoch {epoch:3d}/{epochs}  {phase} {index:3d}/{total}"
        if loss is not None:
            text += f"  loss {loss:.6f}"
        if tty:
            stream.write("\r" + text)
            # Erase on the last iteration so the epoch's own line, printed next, starts
            # at column 0 rather than on top of this one.
            if index >= total:
                stream.write("\r" + " " * len(text) + "\r")
        elif index == 1 or total <= 4 or index % max(1, total // 4) == 0:
            stream.write(f"{stamp()} {text}\n")
        else:
            return
        stream.flush()

    return on_iter


def epoch_logger(log, epochs):
    def on_epoch(entry, is_best):
        line = (f"{stamp()} epoch {entry['epoch']:3d}/{epochs}  "
                f"F1 {entry['val_f1']:.4f}  "
                f"thr {entry['val_threshold']:.2f}  "
                f"sens {entry['val_sensitivity']:.4f}  "
                f"spec {entry['val_specificity']:.4f}  "
                f"ppv {entry['val_ppv']:.4f}  "
                f"auroc {entry['val_auroc']:.4f}  "
                f"loss {entry['train_loss']:.6f}"
                f"{'   <-- NEW BEST' if is_best else ''}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    return on_epoch
