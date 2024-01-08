"""
Run predictions over a list of images

- Visualize predictions
- Calculate the metrics if ground truth is available
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.transforms import get_test_transform
from metrics import DiceScore
from model import init_model
from utils import get_device


def convert_to_image(
    tensor: torch.Tensor, resize_to_wh: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Convert an inout tensor to an image

    Args:
        tensor: CHW tensor float32 range [0.0, 1.0]
        resize_to_wh: Optional resize shape w x h

    Returns:
        Equivalent image HWC uint8 range [0, 255]
    """
    tensor_npy = tensor.cpu().data.numpy()
    image = np.transpose((tensor_npy * 255.0).astype(np.uint8), (1, 2, 0))
    if resize_to_wh:
        image = cv2.resize(image, resize_to_wh)
    return image


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
        "--threshold",
        required=False,
        type=float,
        help="Probability threshold to set 1 as prediction",
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="Model checkpoint filepath"
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="Rescale to the original image size (otherwise keep the model input size)",
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

    batch_size = (
        args.batch_size if args.batch_size is not None else config.train_batch_size
    )
    threshold = args.threshold if args.threshold is not None else config.threshold
    assert threshold is not None

    device = get_device()
    model = init_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    data_transform_test = get_test_transform(config)
    test_dataset = BloodVesselDataset(
        [args.input_path],
        data_transform_test,
        dataset_with_gt=os.path.exists(os.path.join(args.input_path, "labels")),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_batches = len(test_dataset) // config.train_batch_size + int(
        len(test_dataset) % batch_size > 0
    )

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=False)

    dice_score_class = DiceScore(to_monitor=False)

    dice_scores = defaultdict(list)
    predictions = list()

    # Iterate on test batches
    with torch.no_grad():
        for batch_id, data in tqdm(
            enumerate(test_dataloader), desc="batch", total=test_batches
        ):
            # get the input images and labels
            images = data["image"]
            labels = data["label"] if "label" in data else None
            images_paths = data["file"]
            images_shapes = data["shape"]

            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)

            # forward pass to get outputs
            predictions_raw = nn.Sigmoid()(model(images))
            # Prediction is thresholded because rle_encoding used for the challenge
            # requires only 0 or 1 values, while the validation metrics during the training
            # are calculated on the sigmoid raw output
            predictions_thresholded = torch.as_tensor(
                predictions_raw > threshold, dtype=predictions_raw.dtype
            )

            for i in range(images.shape[0]):
                image_path = images_paths[i]

                kidney_slice_name = image_path.split(os.sep)[-3]

                # Shape format is a list of 3 elements (H-W-C).
                # Each element is a list of batch_size length with the values
                original_image_height, original_image_width = (
                    images_shapes[0][i].item(),
                    images_shapes[1][i].item(),
                )
                original_shape_wh = (original_image_width, original_image_height)
                resize_to_wh = original_shape_wh if args.rescale else None

                image = convert_to_image(images[i], resize_to_wh)
                label_img = (
                    convert_to_image(labels[i], resize_to_wh)
                    if labels is not None
                    else None
                )
                predictions_raw_img = convert_to_image(predictions_raw[i], resize_to_wh)
                predictions_thresholded_img = convert_to_image(
                    predictions_thresholded[i], resize_to_wh
                )

                if label_img is not None:
                    # TODO: Extract function(s)
                    diff_on_image = np.copy(image)
                    diff_on_image = cv2.cvtColor(diff_on_image, cv2.COLOR_GRAY2BGR)
                    diff_on_black = np.zeros_like(label_img, np.uint8)
                    diff_on_black = cv2.cvtColor(diff_on_black, cv2.COLOR_GRAY2BGR)

                    # Green
                    correct = (
                        label_img & predictions_thresholded_img
                    )  # np.uint8 of 0 or 1
                    correct_bool = np.squeeze(correct.astype(bool))
                    diff_on_image[correct_bool] = [0, 255, 0]
                    diff_on_black[correct_bool] = [0, 255, 0]

                    # Red
                    false_positive = (label_img == 0) & predictions_thresholded_img
                    false_positive_bool = np.squeeze(false_positive.astype(bool))
                    diff_on_image[false_positive_bool] = [0, 0, 255]
                    diff_on_black[false_positive_bool] = [0, 0, 255]

                    # Blue
                    false_negative = label_img & (predictions_thresholded_img == 0)
                    false_negative_bool = np.squeeze(false_negative.astype(bool))
                    diff_on_image[false_negative_bool] = [255, 0, 0]
                    diff_on_black[false_negative_bool] = [255, 0, 0]

                    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    label_bgr = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)
                    predictions_raw_img_bgr = cv2.cvtColor(
                        predictions_raw_img, cv2.COLOR_GRAY2BGR
                    )
                    predictions_thresholded_img_bgr = cv2.cvtColor(
                        predictions_thresholded_img, cv2.COLOR_GRAY2BGR
                    )

                    row_one = cv2.hconcat([image_bgr, label_bgr, diff_on_image])
                    row_two = cv2.hconcat(
                        [
                            predictions_raw_img_bgr,
                            predictions_thresholded_img_bgr,
                            diff_on_black,
                        ]
                    )
                    all_in_one = cv2.vconcat([row_one, row_two])
                else:
                    all_in_one = cv2.hconcat(
                        [image, predictions_raw_img, predictions_thresholded_img]
                    )
                dice_score = dice_score_class.evaluate(
                    predictions_thresholded[i], labels[i]
                )
                print(f"{image_path} dice_score {dice_score:.2f}")
                dice_scores[kidney_slice_name].append(dice_score)
                predictions.append(
                    {"image_path": image_path, "dice_score": f"{dice_score:.2f}"}
                )
                # TODO: Calculate 3D surface dice metric (target of the competition),
                # but this works only for single kidneys
                cv2.imwrite(
                    os.path.join(
                        args.output_dir,
                        "images",
                        os.path.basename(image_path)[:-4] + f"_dice_score_{dice_score:.2f}.png",
                    ),
                    all_in_one,
                )

    metrics = defaultdict(dict)
    for slice, scores in dice_scores.items():
        avg_dice_score = np.mean(scores)
        print(f"{slice} avg dice score: {avg_dice_score:.2f}")
        metrics[slice]["avg_dice_score"] = f"{avg_dice_score:.2f}"
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as out_fp:
        json.dump(metrics, out_fp, indent=4)
    print(f"Metrics written to {os.path.join(args.output_dir, 'metrics.json')}")

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as out_fp:
        fieldnames = ["image_path", "dice_score"]
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction)
    print(f"Predictions written to {os.path.join(args.output_dir, 'predictions.json')}")


if __name__ == "__main__":
    main()
