import os
import torch
import torch.nn.functional as F


def aggregate(probs):
    return float(probs.mean(axis=0)[1])


def predict(model, dataset, device, conditioned, on_patient=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            images, label, code, views, metadata = dataset[i]
            images = images.to(device)
            if conditioned:
                n = images.shape[0]
                params = {"views": views.to(device).float(),
                          "metadata": metadata.to(device).unsqueeze(0).repeat(n, 1)}
            else:
                params = {}
            probs = F.softmax(model(images, **params), dim=1)
            names = [os.path.basename(p) for p in dataset.images[i]]
            rows.append((code, int(label), aggregate(probs),
                          {name: [float(a), float(b)]
                           for name, (a, b) in zip(names, probs)}))
            if on_patient is not None:
                on_patient(i + 1, len(dataset), rows[-1])
    return rows
