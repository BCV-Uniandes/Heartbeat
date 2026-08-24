from models.baselines import MobileNet, ResNet, VGG16, ViT
from models.heart_vit import HeartViT


def build(arch, *, dim=None, layers=None, dropout=None, heads=None):
    if arch == "heart-vit":
        if dim is None or layers is None:
            raise ValueError("heart-vit needs dim and layers; they vary per fold and "
                             "there is no table left to look them up in")
        return HeartViT(2, view_embedding_dim=dim, adanorm_layers=layers,
                        adanorm_attention_heads=layers if heads is None else heads,
                        dropout=0.0 if dropout is None else dropout,
                        metadata_values=4, pretrained=True), True
    for name, value in (("dim", dim), ("layers", layers), ("dropout", dropout),
                        ("heads", heads)):
        if value is not None:
            raise ValueError(f"{name} applies to heart-vit only, not {arch}")
    builders = {"vit": lambda: ViT(2, pretrained=True),
                "resnet18": lambda: ResNet(2, resnet_type="resnet18"),
                "resnet50": lambda: ResNet(2, resnet_type="resnet50"),
                "vgg16": lambda: VGG16(2),
                "mobilenet": lambda: MobileNet(2)}
    return builders[arch](), False
