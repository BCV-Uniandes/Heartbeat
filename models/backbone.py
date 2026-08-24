from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import (
    EncoderBlock,
    VisionTransformer,
    ViT_B_16_Weights,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaNormEncoderBlock(EncoderBlock):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                         norm_layer)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True))

    def forward(self, input: torch.Tensor, metadata: torch.Tensor):
        torch._assert(input.dim() == 3,
                      f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(metadata).chunk(6, dim=1)

        x = self.ln_1(input)
        x = modulate(x, shift_msa, scale_msa)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = x * gate_msa.unsqueeze(1)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = modulate(y, shift_mlp, scale_mlp)
        y = self.mlp(y) * gate_mlp.unsqueeze(1)
        return x + y


class HeartViTEncoder(nn.Module):
    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim,
                 dropout, attention_dropout, adanorm_layers, adanorm_attention_heads,
                 norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        if hidden_dim % adanorm_attention_heads:
            raise ValueError(
                f"adanorm_attention_heads={adanorm_attention_heads} does not divide "
                f"hidden_dim={hidden_dim}; without this check the mismatch surfaces "
                "later inside F.multi_head_attention_forward")

        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)

        plain: "OrderedDict[str, nn.Module]" = OrderedDict()
        for i in range(num_layers):
            plain[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        self.layers = nn.Sequential(plain)

        extra: "OrderedDict[str, nn.Module]" = OrderedDict()
        for i in range(adanorm_layers):
            extra[f"adanorm_encoder_layer_{i}"] = AdaNormEncoderBlock(
                adanorm_attention_heads, hidden_dim, mlp_dim, dropout,
                attention_dropout, norm_layer)
        self.adanorm_layers = nn.Sequential(extra)

        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, metadata: torch.Tensor):
        torch._assert(input.dim() == 3,
                      f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input + self.pos_embedding)
        x = self.layers(x)
        x = self.dropout(x)
        for block in self.adanorm_layers:
            x = block(x, metadata)
        return self.ln(x)


class HeartViTBackbone(VisionTransformer):
    def __init__(self, *, image_size, patch_size, num_layers, num_heads, hidden_dim,
                 mlp_dim, dropout, attention_dropout, adanorm_layers,
                 adanorm_attention_heads, num_classes):
        super().__init__(image_size=image_size, patch_size=patch_size,
                         num_layers=num_layers, num_heads=num_heads,
                         hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout,
                         attention_dropout=attention_dropout, num_classes=num_classes)
        self.encoder = HeartViTEncoder(
            seq_length=self.seq_length, num_layers=num_layers, num_heads=num_heads,
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout,
            attention_dropout=attention_dropout, adanorm_layers=adanorm_layers,
            adanorm_attention_heads=adanorm_attention_heads)

    def forward(self, x: torch.Tensor):
        raise TypeError(
            "HeartViTBackbone has no usable forward: its encoder is conditioned on "
            "patient metadata, and this signature cannot carry it. Use "
            "models.heart_vit.HeartViT(...)(x, views=..., metadata=...), which "
            "drives _process_input, encoder and heads itself.")


def build_backbone(*, dropout, adanorm_layers, adanorm_attention_heads,
                   pretrained=True):
    model = HeartViTBackbone(
        image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768,
        mlp_dim=3072, dropout=dropout, attention_dropout=0.0,
        adanorm_layers=adanorm_layers,
        adanorm_attention_heads=adanorm_attention_heads, num_classes=1000)
    if pretrained:
        model.load_state_dict(
            ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True), strict=False)
    return model
