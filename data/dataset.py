import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BloodVesselDataset(Dataset):
    def __init__(self, selected_dirs, transform, dataset_with_gt):
        self.selected_dirs = selected_dirs
        self.transform = transform
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
            # Add the channel dimension to the label
            if len(transformed["mask"].shape) == 2:
                transformed["mask"] = torch.unsqueeze(transformed["mask"], dim=0)
            # Additional image only transformations
            return {
                "image": transformed["image"],
                "label": transformed["mask"],
                "file": image_path,
                "shape": list(image.shape),
            }

        else:
            # Transform image
            transformed = self.transform(image=image)
            return {
                "image": transformed["image"],
                "file": image_path,
                "shape": list(image.shape),
            }
