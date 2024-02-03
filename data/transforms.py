import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from config import ConfigParams


def get_train_transform(config: ConfigParams):
    if config.train_augmentation == "2.5d_aug":
        # Augmentations from https://www.kaggle.com/code/yoyobar/2-5d-cutting-model-baseline-training/notebook
        # but this requires a specific TTA or resolution because the crop is always zoomed in.
        return A.Compose(
            [
                # 2.5d augmentation
                A.Rotate(limit=45, p=0.5),
                # Always zoomed in by upscaling ~2x + random crop
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
    elif config.train_augmentation == "my_2.5d_aug":
        # Augmentations from https://www.kaggle.com/code/yoyobar/2-5d-cutting-model-baseline-training/notebook
        # but this requires a specific TTA or resolution because the crop is always zoomed in.
        # Added Invert augmentation.
        return A.Compose(
            [
                # 2.5d augmentation
                A.Rotate(limit=45, p=0.5),
                # Always zoomed in with random scale ~2x (50%) or crop (100%)
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
                A.Rotate(limit=180, p=1.0),
                # Zoom level similar to validation, no TTA strictly necessary, but it helps
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
                    p=1.0,
                ),
                # This is applied only to the image
                A.InvertImg(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GridDistortion(p=0.5),
                A.ToFloat(max_value=255),
                ToTensorV2(),
            ]
        )
    elif config.train_augmentation == "my_aug_v3":
        return A.Compose(
            [
                A.Rotate(
                    limit=180,
                    interpolation=cv2.INTER_NEAREST,
                    rotate_method="largest_box",
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                # Zoom by resizing the image to
                # [0.75 * image_size, 1.25 * image_size] + crop or full resize to model input size
                # Reference default value: v1d image size 910x1303 and input size 512x512
                # 0.75 image size = 683x977 ~ 1.33x,1.9x crop size
                # 1.25 image size = 1138x1628 ~ 2.2x,3.2x crop size
                # so the crop of 512x512 could be
                A.RandomScale(
                    scale_limit=(-0.25, 0.25),
                    interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                # Crop (3x probability of being applied) or full image to match tta at eval time
                A.OneOf(
                    [
                        A.RandomCrop(
                            config.model_train_input_size,
                            config.model_train_input_size,
                            p=3.0
                        ),
                        A.Resize(
                            config.model_train_input_size,
                            config.model_train_input_size,
                            interpolation=cv2.INTER_NEAREST,
                            p=1.0
                        ),
                    ],
                    p=1.0
                ),
                # This is applied only to the image
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    brightness_by_max=True,
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GridDistortion(interpolation=cv2.INTER_NEAREST, normalized=True),
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


def get_test_transform(input_size_height: int, input_size_width: int):
    # Assuming all test images are of the same size, if we want different resize for different images
    # we need to change the logic and resize it in the dataset itself
    return A.Compose(
        [
            A.Resize(
                input_size_height, input_size_width, interpolation=cv2.INTER_NEAREST
            ),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
    )
