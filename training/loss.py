import numpy as np
import torch
import torch.nn.functional as F


def class_weights(samples_per_class, beta=0.9999):
    counts = np.asarray(samples_per_class, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError(
            f"class counts must all be positive, got {list(samples_per_class)}. A zero "
            "count means the split being trained on contains no examples of that class")
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(samples_per_class)
    return torch.tensor(weights, dtype=torch.float64)


def cb_loss(logits, labels, samples_per_class, beta=0.9999):
    no_of_classes = len(samples_per_class)
    weights = class_weights(samples_per_class, beta=beta).to(
        device=logits.device, dtype=logits.dtype)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    sample_weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    sample_weights = sample_weights.sum(1)
    sample_weights = sample_weights.unsqueeze(1)
    sample_weights = sample_weights.repeat(1, no_of_classes)

    return F.binary_cross_entropy_with_logits(
        input=logits, target=labels_one_hot, weight=sample_weights)
