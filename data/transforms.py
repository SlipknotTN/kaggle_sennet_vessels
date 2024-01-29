import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import ConfigParams


def get_train_transform(config: ConfigParams):
    if config.train_augmentation == "2.5d":
        # Augmentations from https://www.kaggle.com/code/yoyobar/2-5d-cutting-model-baseline-training/notebook
        # but this requires a specific TTA because the crop is always zoomed in
        return A.Compose(
            [
                # 2.5d augmentation
                A.Rotate(limit=45, p=0.5),
                # Always zoomed id
                A.RandomScale(scale_limit=(0.8, 1.25), p=0.5),
                A.RandomCrop(
                    config.model_train_input_size, config.model_train_input_size, p=1
                ),
                A.RandomGamma(p=0.75),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.5),
                A.MotionBlur(p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.ToFloat(max_value=255),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif config.train_augmentation == "overfit":
        # Overfit test
        return A.Compose(
            [
                A.Resize(config.model_train_input_size, config.model_train_input_size),
                A.ToFloat(max_value=255),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif config.train_augmentation == "my_aug_v2":
        return A.Compose(
            [
                A.Rotate(limit=180, always_apply=True),
                # Zoom level similar to validation, no TTA necessary?
                A.Resize(config.model_train_input_size, config.model_train_input_size),
                # This is applied only to the image
                A.RandomBrightnessContrast(
                    brightness_limit=0.33,
                    contrast_limit=0.33,
                    brightness_by_max=True,
                    always_apply=True,
                ),
                # # This is applied only to the image
                A.InvertImg(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GridDistortion(p=0.5),
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ]
        )
    else:
        raise Exception(f"Augmentation {config.train_augmentation} not suppported")


def get_val_transform(config: ConfigParams):
    return A.Compose(
        [
            A.Resize(config.model_train_input_size, config.model_train_input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )


def get_test_transform(config: ConfigParams, inference_input_size=None):
    input_size = (
        config.model_train_input_size
        if inference_input_size is None
        else inference_input_size
    )
    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
