import os
import random

import numpy as np
import torch

from evaluation import predict
from dataloader.utils import class_counts
from metrics import evaluate
from training.loss import cb_loss


def select_best(val_f1s):
    if not val_f1s:
        raise ValueError("no epochs to select from; a run with no epochs has no best")
    best_index, best_f1 = 0, val_f1s[0]
    for i, f1 in enumerate(val_f1s):
        if f1 >= best_f1:
            best_index, best_f1 = i, f1
    return best_index


def validate(model, val_dataset, device, conditioned, on_patient=None):
    rows = predict(model, val_dataset, device=device, conditioned=conditioned,
                   on_patient=on_patient)
    labels = [label for _code, label, _prob, _per_image in rows]
    probs = [prob for _code, _label, prob, _per_image in rows]
    return evaluate(labels, probs), rows


def train_fold(model, train_loader, val_dataset, *, epochs, lr, device, conditioned,
               seed, samples_per_class=None, config=None, on_epoch=None,
               on_iter=None, on_val=None, state_path=None, best_path=None,
               weight_decay=0.0, decoupled_weight_decay=True):
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if samples_per_class is None:
        dataset = train_loader.dataset
        if not hasattr(dataset, "labels"):
            raise TypeError(
                f"cannot compute class counts from {type(dataset).__name__}: it has no "
                "`labels`. Pass samples_per_class=[n_no_chd, n_chd] explicitly")
        samples_per_class = class_counts(dataset)
    samples_per_class = [int(n) for n in samples_per_class]

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  decoupled_weight_decay=decoupled_weight_decay)

    resolved = dict(config or {})
    resolved.update({
        "epochs": int(epochs),
        "lr": float(lr),
        "seed": seed,
        "samples_per_class": samples_per_class,
        "batch_size": train_loader.batch_size,
        "train_images": len(train_loader.dataset),
        "val_patients": len(val_dataset),
        "conditioned": bool(conditioned),
        "device": str(device),
        "optimizer": (f"RAdam(weight_decay={weight_decay:g}, "
                      f"decoupled_weight_decay={bool(decoupled_weight_decay)})"),
        "scheduler": None,
        "sampler": type(train_loader.sampler).__name__,
        "samples_per_epoch": len(train_loader.sampler),
    })
    resolved.setdefault("adanorm_attention_heads", None)
    resolved.setdefault("dropout", None)

    entries = []
    best_f1_so_far, best_epoch_so_far = None, None
    best_state, best_rows = None, None
    start_epoch = 1
    if state_path is not None and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["config"] != resolved:
            differing = sorted(k for k in set(state["config"]) | set(resolved)
                               if state["config"].get(k) != resolved.get(k))
            raise ValueError(
                f"{state_path} was written by a run with a different configuration "
                f"({', '.join(differing)}); resuming would splice two runs together. "
                f"Use a different --out, or delete that file to start over")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        entries = state["entries"]
        best_f1_so_far = state["best_f1"]
        best_epoch_so_far = state["best_epoch"]
        best_rows = state["best_rows"]
        if best_path is None or not os.path.exists(best_path):
            raise FileNotFoundError(
                f"{state_path} says epoch {state['best_epoch']} is the best so far, but "
                f"its weights are not at {best_path}. The two are written together and "
                f"one has been removed; resuming would finish by saving whatever the "
                f"last epoch happened to leave in the model and calling it the best")
        best_state = torch.load(best_path, map_location="cpu", weights_only=True)
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        total_iters = len(train_loader)
        for index, (images, labels, views, metadata) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if conditioned:
                params = {"views": views.to(device).float(),
                          "metadata": metadata.to(device)}
            else:
                params = {}
            loss = cb_loss(model(images, **params), labels, samples_per_class)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
            seen += images.shape[0]
            if on_iter is not None:
                on_iter(epoch, index, total_iters, running_loss / seen)

        if not seen:
            raise ValueError(
                "the training loader yielded no samples; the split being trained on is "
                f"empty ({len(train_loader.dataset)} images in its dataset)")

        def report_val(index, total, _row):
            on_val(epoch, index, total)

        scored, rows = validate(model, val_dataset, device, conditioned,
                                report_val if on_val is not None else None)
        f1 = scored["f1"]
        entry = {"epoch": epoch, "train_loss": running_loss / seen,
                 **{f"val_{k}": v for k, v in scored.items()}, "selected": False}
        entries.append(entry)

        is_best = best_f1_so_far is None or f1 >= best_f1_so_far
        if is_best:
            best_f1_so_far, best_epoch_so_far = f1, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_rows = rows
            if best_path is not None:
                torch.save(best_state, best_path)
        if state_path is not None:
            torch.save({"epoch": epoch, "config": resolved, "entries": entries,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "best_f1": best_f1_so_far, "best_epoch": best_epoch_so_far,
                        "best_rows": best_rows, "rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None),
                        "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state()}, state_path)
        if on_epoch is not None:
            on_epoch(entry, is_best)

    best_index = select_best([e["val_f1"] for e in entries])
    entries[best_index]["selected"] = True
    if best_epoch_so_far != entries[best_index]["epoch"]:
        raise RuntimeError(
            f"checkpoint selection disagrees with itself: kept the weights of epoch "
            f"{best_epoch_so_far}, but the record selects epoch "
            f"{entries[best_index]['epoch']}")
    model.load_state_dict(best_state)

    best = entries[best_index]
    return {"config": resolved, "epochs": entries,
            "best": {"epoch": best["epoch"],
                      **{k: v for k, v in best.items()
                         if k.startswith("val_")}},
            "best_rows": best_rows}
