from torch.utils.data import WeightedRandomSampler

from dataloader.dataset import CLASSES


def class_counts(dataset):
    counts = {name: 0 for name in CLASSES}
    for label in dataset.labels:
        counts[CLASSES[label]] += 1
    return [counts[name] for name in CLASSES]


def balanced_sampler(dataset, generator=None):
    counts = class_counts(dataset)
    weights = [1.0 / counts[label] for label in dataset.labels]
    return WeightedRandomSampler(weights, len(dataset), replacement=True,
                                  generator=generator)
