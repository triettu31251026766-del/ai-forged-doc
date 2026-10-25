import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchFeatureExtractor(nn.Module):
    """ResNet18 truncate: dung layer2 + layer3 (upsample ve cung do phan giai)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2,
        )
        self.layer3 = m.layer3
        self.out_channels = 128 + 256  # layer2 + layer3

    def forward(self, x):
        f2 = self.stem(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


class DinoFeatureExtractor(nn.Module):
    """DINOv2 ViT-S/14: dac trung giau ngu nghia, tot hon cho chu/noi dung."""

    def __init__(self, name="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        import timm
        self.model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.out_channels = self.model.num_features

    def forward(self, x):
        feats = self.model.forward_features(x)  # [B, N, C] (co CLS)
        feats = feats[:, 1:, :]                 # bo token CLS
        b, n, c = feats.shape
        h = w = int(n ** 0.5)
        return feats.transpose(1, 2).reshape(b, c, h, w)


def build_feature_extractor(name="resnet18"):
    if name == "dinov2":
        return DinoFeatureExtractor()
    return PatchFeatureExtractor(pretrained=True)


def features_to_patches(feats):
    """feats [B,C,h,w] -> [B*h*w, C] (da L2-normalize)."""
    b, c, h, w = feats.shape
    f = feats.permute(0, 2, 3, 1).reshape(-1, c)
    return F.normalize(f, dim=1), (h, w)


def patch_distances(patches, bank, chunk=8192):
    """patches [P,C], bank [M,C] (da normalize) -> [P] cosine distance (1 - max sim)."""
    best = torch.full((patches.shape[0],), -1.0, device=patches.device)
    for i in range(0, bank.shape[0], chunk):
        sim = patches @ bank[i:i + chunk].t()
        best = torch.maximum(best, sim.max(dim=1).values)
    return 1.0 - best


def score_map(model, x, bank, out_size):
    feats = model(x)
    patches, (h, w) = features_to_patches(feats)
    dist = patch_distances(patches, bank).reshape(1, 1, h, w)
    return F.interpolate(dist, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
