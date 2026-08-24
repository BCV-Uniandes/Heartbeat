import torchvision.transforms as tvt

transform_val = tvt.Compose([
    tvt.ToTensor(),
    tvt.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    tvt.Resize((224, 224), antialias=True),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_train = tvt.Compose([
    tvt.ToTensor(),
    tvt.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    tvt.Resize((256, 256), antialias=True),
    tvt.RandomCrop((224, 224)),
    tvt.RandomHorizontalFlip(),
    tvt.RandomRotation(degrees=10),
    tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
