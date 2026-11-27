import segmentation_models_pytorch as smp


def build_segmentation(encoder="resnet18", encoder_weights="imagenet", in_channels=3, classes=1, arch=None):
    """U-Net by default; SegFormer when arch='segformer' or encoder starts with 'mit_'."""
    if arch == "segformer" or str(encoder).startswith("mit_"):
        return smp.Segformer(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
