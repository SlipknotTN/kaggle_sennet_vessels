import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from config import ConfigParams


def get_train_transform(config: ConfigParams):
    if config.train_augmentation == "my_2.5d_aug":
        # Augmentations from https://www.kaggle.com/code/yoyobar/2-5d-cutting-model-baseline-training/notebook
        # but this requires a specific TTA because the crop is always zoomed in. Added Invert augmentation.
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
                A.InvertImg(p=0.5),
                A.ToFloat(max_value=255),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif config.train_augmentation == "overfit":
        # Overfit test
        return A.Compose(
            [
                A.Resize(
                    config.model_train_input_size,
                    config.model_train_input_size,
                    interpolation=cv2.INTER_NEAREST,
                ),
                A.ToFloat(max_value=255),
                ToTensorV2(transpose_mask=True),
            ]
        )
    elif config.train_augmentation == "my_aug_v2":
        return A.Compose(
            [
                A.Rotate(limit=180, always_apply=True),
                # Zoom level similar to validation, no TTA strictly necessary, but it helps for the score
                A.Resize(
                    config.model_train_input_size,
                    config.model_train_input_size,
                    interpolation=cv2.INTER_NEAREST,
                ),
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
            A.Resize(
                config.model_train_input_size,
                config.model_train_input_size,
                interpolation=cv2.INTER_NEAREST,
            ),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )


def get_test_transform(input_size: int):
    return A.Compose(
        [
            A.Resize(input_size, input_size, interpolation=cv2.INTER_NEAREST),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
