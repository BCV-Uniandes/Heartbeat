import torch
import torch.nn as nn

from models.backbone import build_backbone


class HeartViT(nn.Module):
    def __init__(self, num_classes, *, view_embedding_dim, adanorm_layers,
                 adanorm_attention_heads, dropout, metadata_values=4,
                 pretrained=True):
        super().__init__()
        hidden_dim = 768
        self.backbone = build_backbone(
            dropout=dropout, adanorm_layers=adanorm_layers,
            adanorm_attention_heads=adanorm_attention_heads, pretrained=pretrained)
        # +1 for the view index concatenated onto the metadata vector
        self.metadata_projection = nn.Linear(metadata_values + 1, hidden_dim)
        self.backbone.heads.head = nn.Linear(hidden_dim, num_classes)

    def process_image_features(self, x, metadata):
        x = self.backbone._process_input(x)

        n = x.shape[0]
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x, metadata=metadata)
        return x

    def forward(self, x, *, views, metadata):
        md = torch.cat((views.unsqueeze(1), metadata), dim=1)
        metadata_proj = self.metadata_projection(md)
        class_token = self.process_image_features(x, metadata_proj)[:, 0]
        return self.backbone.heads(class_token)
