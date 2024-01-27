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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.rle import rle_encode
from data.transforms import get_test_transform
from metrics.fast_surface_dice.fast_surface_dice import compute_surface_dice_score
from metrics.metrics import DiceScore, Metric
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
    assert tensor.max() <= 1.0 and tensor.min() >= 0.0
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
        "--batch_size",
        required=False,
        type=int,
        help="Evaluation batch size, otherwise training one is used",
    )
    parser.add_argument(
        "--inference_input_size",
        required=False,
        type=int,
        help="Evaluation input size, otherwise training one is used",
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
        help="Rescale to the original image size (otherwise keep the model input size) "
        "for prediction visualization as images",
    )
    parser.add_argument(
        "--input_paths",
        required=True,
        type=str,
        nargs="*",
        help="List of input directories, each one should correspond to a kidney and "
             "must contain images dir and optionally labels dir",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory with predictions and metrics",
    )
    parser.add_argument(
        "--save_3d_predictions",
        action="store_true",
        help="Save 3d prediction for each dataset subset kidney",
    )
    parser.add_argument(
        "--save_3d_labels",
        action="store_true",
        help="Save 3d label for each dataset subset kidney",
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
    model, preprocess_function, inverse_preprocess_function = init_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)

    data_transform_test = get_test_transform(config, args.inference_input_size)
    labels_exists = [os.path.exists(os.path.join(input_path, "labels")) for input_path in args.input_paths]
    labels_exists_check = np.all(labels_exists)
    print(f"Labels are not available for at least one of {args.input_paths}")
    test_dataset = BloodVesselDataset(
        args.input_paths,
        data_transform_test,
        preprocess_function=preprocess_function,
        dataset_with_gt=labels_exists_check,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_batches = len(test_dataset) // config.train_batch_size + int(
        len(test_dataset) % batch_size > 0
    )

    kidney_subset_names = [os.path.basename(input_path) for input_path in args.input_paths]

    for kidney_subset_name in kidney_subset_names:
        os.makedirs(os.path.join(args.output_dir, f"images_{kidney_subset_name}"), exist_ok=False)
    os.makedirs(os.path.join(args.output_dir, "3d_npy"), exist_ok=False)

    dice_score_class: Metric = DiceScore(to_monitor=False)

    dice_2d_scores = defaultdict(list)
    single_2d_slices_scores = list()

    # Keep track of all the predictions for 3D evaluation
    predictions_for_3d = defaultdict(list)
    labels_for_3d = defaultdict(list)

    predictions_df_dict = dict()
    labels_df_dict = dict()

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
            if labels is not None:
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

                dataset_kidney_name = image_path.split(os.sep)[-3]

                # Shape format is a list of 3 elements (H-W-C).
                # Each element is a list of batch_size length with the values
                original_image_height, original_image_width = (
                    images_shapes[0][i].item(),
                    images_shapes[1][i].item(),
                )
                original_shape_wh = (original_image_width, original_image_height)
                resize_to_wh = original_shape_wh if args.rescale else None

                # Restore original image values range
                original_image = inverse_preprocess_function(x_norm=images[i])
                image = convert_to_image(original_image, resize_to_wh)
                label_img = (
                    convert_to_image(labels[i], resize_to_wh)
                    if labels is not None
                    else None
                )

                prediction_raw_img = convert_to_image(predictions_raw[i], resize_to_wh)
                prediction_thresholded_img = convert_to_image(
                    predictions_thresholded[i], resize_to_wh
                )

                # Calculate prediction and labels upscaled for 3D metrics and 3D results export (only 0-1 values)
                # the fast_surface_dice implementation requires two dataframes as input: prediction and label

                # Manipulate prediction with torch, 4D input is necessary for upsampling,
                # upsampling is made at the original image size
                prediction = torch.unsqueeze(predictions_thresholded[i], dim=0)  # 1 x 1 x height x width shape
                # print(f"Prediction shape: {prediction.shape}")
                # print(f"Original shape w x h: {original_image_width} x {original_image_height}")
                prediction_upscaled_th = nn.UpsamplingNearest2d(
                    size=[original_image_height, original_image_width]
                )(prediction)
                prediction_upscaled_th = torch.squeeze(prediction_upscaled_th)
                # Numpy shape HW for rle encode
                prediction_upscaled_npy = prediction_upscaled_th.cpu().data.detach().numpy(force=True)
                prediction_rle = rle_encode(np.squeeze(prediction_upscaled_npy))
                prediction_row_df = pd.DataFrame.from_dict(
                    {
                        "id": [
                            f"{dataset_kidney_name}_{os.path.basename(image_path[:-4])}"
                        ],
                        "rle": [prediction_rle],
                        "width": [original_image_width],  # Not mandatory
                        "height": [original_image_height],  # Not mandatory
                        "image_id": [dataset_kidney_name],  # Not mandatory
                        "slice_id": [
                            os.path.basename(image_path[:-4]).split("_")[-1]
                        ],  # Not mandatory
                    }
                )
                if dataset_kidney_name not in predictions_df_dict:
                    predictions_df_dict[dataset_kidney_name] = pd.DataFrame(
                        columns=["id", "rle", "width", "height", "image_id", "slice_id"]
                    )
                predictions_df_dict[dataset_kidney_name] = pd.concat(
                    [predictions_df_dict[dataset_kidney_name], prediction_row_df],
                    ignore_index=True,
                )
                predictions_for_3d[dataset_kidney_name].append(
                    prediction_upscaled_npy.astype(bool)
                )

                if labels is None:
                    all_in_one = cv2.hconcat(
                        [image, prediction_raw_img, prediction_thresholded_img]
                    )
                    cv2.imwrite(
                        os.path.join(
                            args.output_dir,
                            f"images_{dataset_kidney_name}",
                            os.path.basename(image_path)
                        ),
                        all_in_one,
                    )
                else:
                    # CHW
                    label_npy = labels[i].cpu().data.numpy()
                    # HWC
                    label_npy = np.transpose(label_npy, (1, 2, 0))
                    label_upscaled = cv2.resize(
                        label_npy, original_shape_wh, interpolation=cv2.INTER_NEAREST
                    )
                    # WH for rle encode
                    label_rle = rle_encode(np.squeeze(label_upscaled))
                    label_row_df = pd.DataFrame.from_dict(
                        {
                            "id": [
                                f"{dataset_kidney_name}_{os.path.basename(image_path[:-4])}"
                            ],
                            "rle": [label_rle],
                            "width": [original_image_width],
                            "height": [original_image_height],
                            "image_id": [dataset_kidney_name],
                            "slice_id": [os.path.basename(image_path[:-4]).split("_")[-1]],
                        }
                    )
                    if dataset_kidney_name not in labels_df_dict:
                        labels_df_dict[dataset_kidney_name] = pd.DataFrame(
                            columns=["id", "rle", "width", "height", "image_id", "slice_id"]
                        )
                    labels_df_dict[dataset_kidney_name] = pd.concat(
                        [labels_df_dict[dataset_kidney_name], label_row_df],
                        ignore_index=True,
                    )
                    labels_for_3d[dataset_kidney_name].append(label_upscaled.astype(bool))

                    # TODO: Extract function(s)
                    diff_on_image = np.copy(image)
                    diff_on_image = cv2.cvtColor(diff_on_image, cv2.COLOR_GRAY2BGR)
                    diff_on_black = np.zeros_like(label_img, np.uint8)
                    diff_on_black = cv2.cvtColor(diff_on_black, cv2.COLOR_GRAY2BGR)

                    # Green
                    correct = (
                        label_img & prediction_thresholded_img
                    )  # np.uint8 of 0 or 1
                    correct_bool = np.squeeze(correct.astype(bool))
                    diff_on_image[correct_bool] = [0, 255, 0]
                    diff_on_black[correct_bool] = [0, 255, 0]

                    # Red
                    false_positive = (label_img == 0) & prediction_thresholded_img
                    false_positive_bool = np.squeeze(false_positive.astype(bool))
                    diff_on_image[false_positive_bool] = [0, 0, 255]
                    diff_on_black[false_positive_bool] = [0, 0, 255]

                    # Blue
                    false_negative = label_img & (prediction_thresholded_img == 0)
                    false_negative_bool = np.squeeze(false_negative.astype(bool))
                    diff_on_image[false_negative_bool] = [255, 0, 0]
                    diff_on_black[false_negative_bool] = [255, 0, 0]

                    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    label_bgr = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)
                    predictions_raw_img_bgr = cv2.cvtColor(
                        prediction_raw_img, cv2.COLOR_GRAY2BGR
                    )
                    predictions_thresholded_img_bgr = cv2.cvtColor(
                        prediction_thresholded_img, cv2.COLOR_GRAY2BGR
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

                    dice_score = dice_score_class.evaluate(
                        predictions_thresholded[i], labels[i]
                    )
                    print(f"{image_path} {dice_score_class.name} {dice_score:.2f}")
                    dice_2d_scores[dataset_kidney_name].append(dice_score)
                    single_2d_slices_scores.append(
                        {
                            "image_path": image_path,
                            dice_score_class.name: f"{dice_score:.2f}",
                        }
                    )
                    cv2.imwrite(
                        os.path.join(
                            args.output_dir,
                            f"images_{dataset_kidney_name}",
                            os.path.basename(image_path)[:-4]
                            + f"_{dice_score_class.name}_{dice_score:.2f}.png",
                        ),
                        all_in_one,
                    )

    if labels_exists_check:
        metrics = defaultdict(dict)
        # 2D metrics: average 2D dice score (on each 2D slice)
        for dataset_kidney_name, scores in dice_2d_scores.items():
            avg_2d_dice_score = np.mean(scores)
            print(f"{dataset_kidney_name} avg 2D dice score: {avg_2d_dice_score:.2f}")
            metrics[dataset_kidney_name]["avg_2D_dice_score"] = f"{avg_2d_dice_score:.2f}"

        print(f"{dataset_kidney_name}: calculating surface dice score...")
        surface_dice_score = compute_surface_dice_score(
            predictions_df_dict[dataset_kidney_name],
            labels_df_dict[dataset_kidney_name],
            device,
        )
        print(f"Surface dice score:  {surface_dice_score}")
        metrics[dataset_kidney_name]["surface_dice_score"] = surface_dice_score
        # Save also the dataframes
        predictions_df_dict[dataset_kidney_name].to_csv(
            os.path.join(
                args.output_dir, f"slices_predictions_rle_{dataset_kidney_name}.csv"
            ),
            index=False,
            sep=";",
        )
        labels_df_dict[dataset_kidney_name].to_csv(
            os.path.join(
                args.output_dir, f"slices_labels_rle_{dataset_kidney_name}.csv"
            ),
            index=False,
            sep=";",
        )

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as out_fp:
            json.dump(metrics, out_fp, indent=4)
        print(f"Metrics written to {os.path.join(args.output_dir, 'metrics.json')}")

        with open(
                os.path.join(args.output_dir, "single_2d_slices_scores.csv"), "w"
        ) as out_fp:
            fieldnames = ["image_path", dice_score_class.name]
            writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
            writer.writeheader()
            for single_2d_slice_scores_row in single_2d_slices_scores:
                writer.writerow(single_2d_slice_scores_row)
        print(
            f"2D slices scores written to {os.path.join(args.output_dir, 'single_2d_slices_scores.csv')}"
        )

    # 3D metrics: surface dice and save 3D numpy array for later analysis
    for dataset_kidney_name in predictions_for_3d.keys():
        print(f"{dataset_kidney_name}: saving 3d numpy prediction and label...")
        if args.save_3d_predictions:
            kidney_predictions_for_3d = predictions_for_3d[dataset_kidney_name]
            kidney_prediction_3d = np.stack(kidney_predictions_for_3d, axis=0)  # CHW
            np.save(
                os.path.join(
                    args.output_dir, "3d_npy", f"{dataset_kidney_name}_prediction.npy"
                ),
                kidney_prediction_3d,
            )
        if args.save_3d_labels:
            if labels_exists_check:
                kidney_labels_for_3d = labels_for_3d[dataset_kidney_name]
                kidney_label_3d = np.stack(kidney_labels_for_3d, axis=0)  # CHW
                np.save(
                    os.path.join(
                        args.output_dir, "3d_npy", f"{dataset_kidney_name}_label.npy"
                    ),
                    kidney_label_3d,
                )
            else:
                print("Labels not available for all the input directories, saving 3D labels will be skipped")


if __name__ == "__main__":
    main()
