"""
Run predictions over a list of images

- Visualize predictions
- Calculate the metrics if ground truth is available
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.transforms import get_test_transform
from model import UnetModel
from utils import get_device


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Eval the model",
    )
    parser.add_argument(
        "--config_path", required=True, type=str, help="Configuration filepath"
    )
    parser.add_argument(
        "--batch_size", required=False, type=int, help="Evaluation batch size"
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Model checkpoint filepath"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="Must contain images dir and optionally labels dir",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory with predictions and metrics",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    config = ConfigParams(args.config_path)

    device = get_device()
    model = UnetModel()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    data_transform_test = get_test_transform(config)
    test_dataset = BloodVesselDataset(
        [args.input_path],
        data_transform_test,
        dataset_with_gt=os.path.exists(os.path.join(args.input_path, "labels")),
    )
    batch_size = args.batch_size if args.batch_size else config.train_batch_size
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_batches = len(test_dataset) // config.train_batch_size + int(
        len(test_dataset) % batch_size > 0
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate on test batches
    for batch_id, data in tqdm(
        enumerate(test_dataloader), desc="batch", total=test_batches
    ):
        # get the input images and labels
        images = data["image"]
        labels = data["label"] if "label" in data else None
        image_paths = data["file"]
        # print(f"Images shape: {images.shape}, labels shape: {labels.shape}")

        # Move to GPU
        images = images.to(device)

        # forward pass to get outputs
        outputs = model(images)

        for i in range(images.shape[0]):
            image_npy = images[i].cpu().data.numpy()
            image_path = image_paths[i]
            label_npy = labels[i].data.numpy() if labels is not None else None
            output_npy = outputs[i].cpu().data.numpy()
            # numpy [0, 255], uint8, HWC (single channel)
            image_img = np.transpose((image_npy * 255.0).astype(np.uint8), (1, 2, 0))
            label_img = np.transpose(
                (
                    (label_npy * 255.0).astype(np.uint8)
                    if label_npy is not None
                    else None
                ),
                (1, 2, 0),
            )
            output_img = np.transpose((output_npy * 255.0).astype(np.uint8), (1, 2, 0))

            if label_img is not None:
                all_in_one = cv2.hconcat([image_img, label_img, output_img])
            else:
                all_in_one = cv2.hconcat([image_img, output_img])
            print(image_path)
            cv2.imwrite(
                os.path.join(
                    args.output_dir, os.path.basename(image_path)[:-4] + ".png"
                ),
                all_in_one,
            )
            # TODO: Create another tile with label-output diff
            # TODO: Implement metrics

        # Why is this necessary to avoid OOM?
        del images
        del outputs


if __name__ == "__main__":
    main()
