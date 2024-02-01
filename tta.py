"""
Predictions are thresholded because rle_encoding used for the challenge requires only 0 or 1 values,
while the validation metrics during the training are calculated on the sigmoid raw output.
"""
from typing import Tuple

import torch


def predict_no_tta(prediction_raw, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    # No test time augmentation
    prediction_thresholded = torch.as_tensor(
        prediction_raw > threshold, dtype=prediction_raw.dtype
    )
    return prediction_raw, prediction_thresholded


def predict_crops_tta_max(
    full_image_prediction_raw,
    tl_prediction_raw,
    tr_prediction_raw,
    bl_prediction_raw,
    br_prediction_raw,
    center_prediction_raw,
    threshold,
    tta_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # One crop: full image
    # Other crops are zoomed in parts: top_left, top_right, bottom_left, bottom_right and center
    assert tta_mode in [
        "4+full_max",
        "5+full_max",
    ], f"tta_mode {tta_mode} not supported"

    # Reference image to put all the predictions resolution: (model_input_size * 2) x (model_input_size * 2)

    # 4 crops to a single output map
    not_overlapping_crops_pred_ref_top = torch.concat(
        [tl_prediction_raw, tr_prediction_raw], dim=2
    )
    not_overlapping_crops_pred_ref_bottom = torch.concat(
        [bl_prediction_raw, br_prediction_raw], dim=2
    )
    not_overlapping_crops_pred_ref = torch.concat(
        [not_overlapping_crops_pred_ref_top, not_overlapping_crops_pred_ref_bottom],
        dim=1,
    )

    # Pad the center crop to match the desired size
    pad_size_height, pad_size_width = int(full_image_prediction_raw.shape[1] / 2), int(
        full_image_prediction_raw.shape[2] / 2
    )
    center_crop_pred_ref = torch.nn.functional.pad(
        center_prediction_raw,
        (pad_size_width, pad_size_width, pad_size_height, pad_size_height),
        mode="constant",
        value=0.0,
    )

    # Upsample the full image prediction: pass to 4D for the operation and then back to 3D
    full_image_pred_ref = torch.squeeze(
        torch.nn.UpsamplingNearest2d(
            size=[
                not_overlapping_crops_pred_ref.shape[1],
                not_overlapping_crops_pred_ref.shape[2],
            ]
        )(torch.unsqueeze(full_image_prediction_raw, dim=0)),
        dim=0,
    )

    # Only max aggregation between different crops and full image is supported: 4 sides crops or 5 including center
    if tta_mode == "5+full_max":
        max_between_crops_pred_ref = torch.maximum(
            not_overlapping_crops_pred_ref, center_crop_pred_ref
        )
    else:
        max_between_crops_pred_ref = not_overlapping_crops_pred_ref

    prediction_raw = torch.maximum(max_between_crops_pred_ref, full_image_pred_ref)

    prediction_thresholded = torch.as_tensor(
        prediction_raw > threshold, dtype=prediction_raw.dtype
    )

    return prediction_raw, prediction_thresholded
