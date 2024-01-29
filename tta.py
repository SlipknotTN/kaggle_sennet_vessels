"""
Predictions are thresholded because rle_encoding used for the challenge requires only 0 or 1 values,
while the validation metrics during the training are calculated on the sigmoid raw output.
"""
import torch
from typing import Tuple


def predict_no_tta(prediction_raw, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    # No test time augmentation
    prediction_thresholded = torch.as_tensor(
        prediction_raw > threshold, dtype=prediction_raw.dtype
    )
    return prediction_raw, prediction_thresholded


def predict_crops_tta_max(full_image_prediction_raw,
                  tl_prediction_raw, tr_prediction_raw, bl_prediction_raw, br_prediction_raw,
                  center_prediction_raw, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    # One crop: full image
    # Other crops are zoomed in parts
    # Aggregation between different crops with max

    # Reference image to put all the predictions at (model_input_size * 2) x (model_input_size * 2)

    # TODO: Test with just crops to check validity
    not_overlapping_crops_pred_top = torch.concat([tl_prediction_raw, tr_prediction_raw], dim=2)
    not_overlapping_crops_pred_bottom = torch.concat([bl_prediction_raw, br_prediction_raw], dim=2)
    not_overlapping_crops_pred = torch.concat([not_overlapping_crops_pred_top, not_overlapping_crops_pred_bottom], dim=1)

    # TODO: Implement the real logic
    prediction_thresholded = torch.as_tensor(
        not_overlapping_crops_pred > threshold, dtype=not_overlapping_crops_pred.dtype
    )

    return not_overlapping_crops_pred, prediction_thresholded
