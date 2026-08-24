import csv, os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataloader.transforms import transform_train, transform_val

VIEW_INDEX = {"3VT": 0, "4C": 1, "RVOT": 2, "LVOT": 3}
METADATA_FIELDS = ["maternal_age", "gestational_age_wk", "three_vessels", "growth_percentile"]

COHORTS = ("2T", "3T")
CLASSES = ("No_CHD", "CHD")

DEFAULT_TRANSFORMS = {"train": transform_train}
TRANSFORM_NAMES = {id(transform_train): "transform_train",
                   id(transform_val): "transform_val"}


class DelfosPatientDataset(Dataset):
    def __init__(self, root, metadata_path, transform=None, patient_codes=None,
                 cohort=None):
        self.root = root
        self.transform = transform if transform is not None else transform_val
        self.classes = list(CLASSES)
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}
        self.cohort = cohort if cohort is not None else self._infer_cohort(root)
        self.metadata_dict = self._read_metadata(metadata_path)
        self.images, self.labels, self.codes, self.views, self.metadata = [], [], [], [], []
        self._load(patient_codes)

    @staticmethod
    def _infer_cohort(root):
        for part in os.path.abspath(root).split(os.sep):
            if part in COHORTS:
                return part
        raise ValueError(f"cannot infer cohort from {root!r}; pass cohort= explicitly")

    def _read_metadata(self, path):
        out = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = row["image"].rsplit("/", 1)[-1]
                out[name] = [float(row[f]) for f in METADATA_FIELDS]
        return out

    def _load(self, patient_codes):
        for class_idx, class_name in enumerate(self.classes):
            cdir = os.path.join(self.root, class_name)
            if not os.path.isdir(cdir):
                continue
            for code in sorted(os.listdir(cdir)):
                if patient_codes is not None and code not in patient_codes:
                    continue
                pdir = os.path.join(cdir, code)
                if not os.path.isdir(pdir):
                    continue
                names = sorted((f for f in os.listdir(pdir) if f.endswith(".png")),
                               key=lambda f: (VIEW_INDEX[f.split(".")[1]], f))
                if not names:
                    continue
                self.images.append([os.path.join(pdir, f) for f in names])
                self.labels.append(class_idx)
                self.codes.append(code)
                self.views.append([VIEW_INDEX[f.split(".")[1]] for f in names])
                self.metadata.append(self.metadata_dict[names[0]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        paths = self.images[idx]
        images = torch.stack([self.transform(np.array(Image.open(p).convert("RGB")))
                              for p in paths])
        views = torch.tensor(self.views[idx], dtype=torch.int64)
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float)
        return images, self.labels[idx], self.codes[idx], views, metadata


class DelfosImageDataset(Dataset):
    def __init__(self, root, metadata_path, fold_csv, split, transform=None):
        self.root = root
        self.transform = (transform if transform is not None
                          else DEFAULT_TRANSFORMS.get(split, transform_val))
        self.transform_name = TRANSFORM_NAMES.get(id(self.transform),
                                                  type(self.transform).__name__)
        self.classes = list(CLASSES)
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}
        self.metadata_dict = DelfosPatientDataset._read_metadata(self, metadata_path)

        self.paths, self.labels, self.views, self.metadata = [], [], [], []
        with open(fold_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] != split:
                    continue
                class_name = row["class"]
                patient_code = row["patient_code"]
                filename = row["image"].rsplit("/", 1)[-1]
                path = os.path.join(root, class_name, patient_code, filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"{fold_csv}: {row['image']!r} resolved to {path!r}, which "
                        "doesn't exist -- fold CSV and directory layout disagree")
                self.paths.append(path)
                self.labels.append(self.class_to_idx[class_name])
                self.views.append(VIEW_INDEX[filename.split(".")[1]])
                self.metadata.append(self.metadata_dict[filename])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(np.array(Image.open(self.paths[idx]).convert("RGB")))
        view = torch.tensor(self.views[idx], dtype=torch.int64)
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float)
        return image, self.labels[idx], view, metadata
