import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BloodVesselDataset(Dataset):
    def __init__(self, selected_dirs, transform, preprocess_function, dataset_with_gt):
        self.selected_dirs = selected_dirs
        self.transform = transform
        self.preprocess_function = preprocess_function
        self.dataset_with_gt = dataset_with_gt
        self.samples = []
        for selected_dir in selected_dirs:
            images_dir = os.path.join(selected_dir, "images")
            assert os.path.exists(images_dir), f"{images_dir} does not exist"
            images_filepaths = sorted(os.listdir(images_dir))
            assert len(images_filepaths) > 0
            labels_filepaths = []
            if dataset_with_gt:
                labels_dir = os.path.join(selected_dir, "labels")
                assert os.path.exists(labels_dir), f"{labels_dir} does not exist"
                labels_filepaths = os.listdir(os.path.join(selected_dir, "labels"))
                assert len(images_filepaths) == len(labels_filepaths), (
                    f"Number of images {len(images_filepaths)} != number of labels {len(labels_filepaths)} "
                    f"for dir {selected_dir}"
                )
            print(
                f"{selected_dir}: images {len(images_filepaths)}, labels {len(labels_filepaths)}"
            )
            for image_filepath in tqdm(images_filepaths, desc="image"):
                full_image_filepath = os.path.join(
                    selected_dir, "images", image_filepath
                )
                if self.dataset_with_gt:
                    full_label_filepath = os.path.join(
                        selected_dir, "labels", image_filepath
                    )
                else:
                    full_label_filepath = None
                self.samples.append([full_image_filepath, full_label_filepath])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)

        if sample[1]:
            label_path = sample[1]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # HW, [0.0, 1.0] range, no channel dimension
            label = label.astype(np.float32) / 255.0
            # Transform image and label
            transformed = self.transform(image=image, mask=label)
            # Add the channel dimension to the label to CHW
            if len(transformed["mask"].shape) == 2:
                transformed["mask"] = torch.unsqueeze(transformed["mask"], dim=0)
            image_preprocessed = self.preprocess_function(transformed["image"])
            return {
                "image": image_preprocessed,
                "label": transformed["mask"],
                "file": image_path,
                "shape": list(image.shape),
            }

        else:
            # Transform image
            transformed = self.transform(image=image)
            image_preprocessed = self.preprocess_function(transformed["image"])
            return {
                "image": image_preprocessed,
                "file": image_path,
                "shape": list(image.shape),
            }


class BloodVesselDatasetTTA(BloodVesselDataset):
    """
    Dataset with test time augmentation support (not implemented with albumentation)
    """

    def __init__(
        self,
        selected_dirs,
        transform,
        preprocess_function,
        dataset_with_gt,
        input_size_width,
        input_size_height,
        tta_mode=None,
    ):
        super().__init__(selected_dirs, transform, preprocess_function, dataset_with_gt)
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        self.tta_mode = tta_mode
        assert (
            self.tta_mode in [None, ""]
            or self.tta_mode.startswith("4+full")
            or self.tta_mode.startswith("5+full")
        ), f"tta_mode {self.tta_mode} not supported"
        if self.tta_mode in [None, ""]:
            self.tta_mode = None
        else:
            assert input_size_width % 2 == 0, f"input_size_width {input_size_width} is not divisible by 2"
            assert input_size_height % 2 == 0, f"input_size_height {input_size_height} is not divisible by 2"

    def __getitem__(self, idx):
        if self.tta_mode is None or self.tta_mode == "":
            # Not TTA
            return super().__getitem__(idx)

        elif self.tta_mode.startswith("4+full") or self.tta_mode.startswith("5+full"):
            # One crop: full image resized at input_size x input_size
            # 4 crops option: 4 crops of input_size x input_size at
            #   top_left, top_right, bottom_left, bottom_right
            #   No overlapping between the 4 crops
            # 5 crops option: 5 crops of input_size x input_size at
            #   top_left, top_right, bottom_left, bottom_right and additional center

            # To create the crops the full image is first resized to (2 * input_size) x (2 * input_size)
            sample = self.samples[idx]

            image_path = sample[0]
            full_image_2dims = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            full_image_3dims = np.expand_dims(full_image_2dims, axis=-1)

            # Get crops
            image_double_size_2dims = cv2.resize(
                full_image_2dims,
                (self.input_size_width * 2, self.input_size_height * 2),
                cv2.INTER_NEAREST,
            )
            image_double_size_3dims = np.expand_dims(image_double_size_2dims, axis=-1)
            top_left_3dims = image_double_size_3dims[
                0 : self.input_size_height, 0 : self.input_size_width, :
            ]
            top_right_3dims = image_double_size_3dims[
                0 : self.input_size_height, self.input_size_width :, :
            ]
            bottom_left_3dims = image_double_size_3dims[
                self.input_size_height :, 0 : self.input_size_width, :
            ]
            bottom_right_3dims = image_double_size_3dims[
                self.input_size_height :, self.input_size_width :, :
            ]
            center_3dims = image_double_size_3dims[
                int(self.input_size_height / 2) : -int(self.input_size_height / 2),
                int(self.input_size_width / 2) : -int(self.input_size_width / 2),
                :,
            ]
            top_left_preprocessed = self.preprocess_function(
                self.transform(image=top_left_3dims)["image"]
            )
            top_right_preprocessed = self.preprocess_function(
                self.transform(image=top_right_3dims)["image"]
            )
            bottom_left_preprocessed = self.preprocess_function(
                self.transform(image=bottom_left_3dims)["image"]
            )
            bottom_right_preprocessed = self.preprocess_function(
                self.transform(image=bottom_right_3dims)["image"]
            )
            # Return center preprocessed even if not used
            center_preprocessed = self.preprocess_function(
                self.transform(image=center_3dims)["image"]
            )

            # Get label and transform it if available
            if sample[1]:
                label_path = sample[1]
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # HW, [0.0, 1.0] range, no channel dimension
                label = label.astype(np.float32) / 255.0
                # Transform full image and label
                transformed_full_image_and_label_dict = self.transform(
                    image=full_image_3dims, mask=label
                )
                # Add the channel dimension to the label to CHW
                if len(transformed_full_image_and_label_dict["mask"].shape) == 2:
                    transformed_full_image_and_label_dict["mask"] = torch.unsqueeze(
                        transformed_full_image_and_label_dict["mask"], dim=0
                    )
                full_image_preprocessed = self.preprocess_function(
                    transformed_full_image_and_label_dict["image"]
                )
                return {
                    "image": full_image_preprocessed,
                    "label": transformed_full_image_and_label_dict["mask"],
                    "file": image_path,
                    "shape": list(full_image_2dims.shape),
                    "top_left": top_left_preprocessed,
                    "top_right": top_right_preprocessed,
                    "bottom_left": bottom_left_preprocessed,
                    "bottom_right": bottom_right_preprocessed,
                    "center": center_preprocessed,
                }
            else:
                # Transform image
                transformed_full_image_dict = self.transform(image=full_image_3dims)
                full_image_preprocessed = self.preprocess_function(
                    transformed_full_image_dict["image"]
                )

                return {
                    "image": full_image_preprocessed,
                    "file": image_path,
                    "shape": list(full_image_2dims.shape),
                    "top_left": top_left_preprocessed,
                    "top_right": top_right_preprocessed,
                    "bottom_left": bottom_left_preprocessed,
                    "bottom_right": bottom_right_preprocessed,
                    "center": center_preprocessed,
                }
        else:
            raise Exception(f"tta_mode {self.tta_mode} not supported")
