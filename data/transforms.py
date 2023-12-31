import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from config import ConfigParams


def get_train_transform(config: ConfigParams):
    return A.Compose(
        # 2.5d augmentation
        [
            A.Rotate(limit=45, p=0.5),
            A.RandomScale(scale_limit=(0.8, 1.25), p=0.5),
            A.RandomCrop(config.model_input_size, config.model_input_size, p=1),
            A.RandomGamma(p=0.75),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ToFloat(max_value=255),
            ToTensorV2(transpose_mask=True),
            # A.Rotate(limit=180, p=0.75),
            # A.Resize(config.train_resize_before_crop, config.train_resize_before_crop),
            # A.RandomScale(0.2, always_apply=True),
            # A.RandomCrop(config.model_input_size, config.model_input_size),
            # # This is applied only to the image
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.33,
            #     contrast_limit=0.33,
            #     brightness_by_max=True,
            #     always_apply=True,
            # ),
            # # This is applied only to the image
            # # TODO: Maybe too much, but at least it predicts something on kidney_3
            # A.InvertImg(p=0.5),
            # #A.HorizontalFlip(p=0.5),
            # #A.VerticalFlip(p=0.5),
            # # A.GridDistortion(p=0.5),
            # # This is applied only to the image
            # A.ToFloat(max_value=255),
            # ToTensorV2(),
        ],
    )


def get_val_transform(config: ConfigParams):
    return A.Compose(
        [
            A.Resize(config.model_input_size, config.model_input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )


def get_test_transform(config: ConfigParams):
    # TODO: Try equalization?!
    return A.Compose(
        [
            A.Resize(config.model_input_size, config.model_input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
