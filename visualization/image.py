from typing import Optional, Tuple

import cv2
import numpy as np
import torch


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
        image = cv2.resize(image, resize_to_wh, cv2.INTER_NEAREST)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image
