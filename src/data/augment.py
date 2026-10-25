import albumentations as A


def build_train_augment():
    """Augmentation manh de giam phu thuoc vao artifact toan anh (do phan giai, nen)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        A.GaussNoise(p=0.25),
        A.ImageCompression(quality_range=(30, 90), p=0.5),
    ])
