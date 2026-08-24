import json
import os


def write_predictions(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    codes = [code for code, _label, _prob, _per_image in rows]
    duplicates = sorted({c for c in codes if codes.count(c) > 1})
    if duplicates:
        raise ValueError(
            f"{path}: duplicate patient code(s) {', '.join(duplicates)} in {len(rows)} "
            f"rows. Keying by code would collapse them and drop a patient from the "
            f"table without any error. Fix the caller's patient list.")
    out = {code: {"label": label, "prob_chd": prob, "images": per_image}
           for code, label, prob, per_image in sorted(rows)}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
