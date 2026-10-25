import torch.nn as nn
from torchvision import models


def build_classifier(name="resnet18", pretrained=True, num_classes=1):
    """Tra ve model voi 1 logit (dung BCEWithLogitsLoss)."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Model khong ho tro: {name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
