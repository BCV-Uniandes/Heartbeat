import torch
import torch.nn as nn
import torchvision

from torchvision.models import ViT_B_16_Weights, vit_b_16


class _FlatBatch(nn.Module):
    def forward(self, x):
        return self.model(x)


class ResNet(_FlatBatch):
    def __init__(self, num_classes, resnet_type="resnet18"):
        super().__init__()
        if resnet_type == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, num_classes, bias=True)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, num_classes, bias=True)


class MobileNet(_FlatBatch):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights="DEFAULT")
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)


class VGG16(_FlatBatch):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.vgg16(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, num_classes)


class ViT(nn.Module):
    def __init__(self, num_classes, *, pretrained=True, dropout=0.0):
        super().__init__()
        hidden_dim = 768
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights, dropout=dropout)
        self.concat_projection = nn.Linear(hidden_dim, hidden_dim)
        self.backbone.heads.head = nn.Linear(hidden_dim, num_classes)

    def _class_token(self, x):
        x = self.backbone._process_input(x)
        n = x.shape[0]
        cls = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.backbone.encoder(x)[:, 0]

    def forward(self, x):
        return self.backbone.heads(self.concat_projection(self._class_token(x)))
