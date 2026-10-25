import segmentation_models_pytorch as smp


def build_segmentation(encoder="resnet18", encoder_weights="imagenet", in_channels=3, classes=1):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
