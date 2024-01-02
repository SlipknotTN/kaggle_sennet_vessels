import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import ConfigParams


def get_train_transform(config: ConfigParams):
    return A.Compose(
        [
            A.RandomScale(0.2, always_apply=True),
            A.Resize(config.model_input_size, config.model_input_size),
            #A.Resize(config.train_resize_before_crop, config.train_resize_before_crop),
            #A.RandomCrop(config.model_input_size, config.model_input_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
        additional_targets={"label": "image"},
    )


def get_val_transform(config: ConfigParams):
    return A.Compose(
        [
            A.Resize(config.model_input_size, config.model_input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
        additional_targets={"label": "image"},
    )


def get_test_transform(config: ConfigParams):
    return A.Compose(
        [
            A.Resize(config.model_input_size, config.model_input_size),
            A.ToFloat(max_value=255),
            ToTensorV2(),
        ],
        additional_targets={"label": "image"},
    )
