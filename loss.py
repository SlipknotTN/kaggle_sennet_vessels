import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.dice import DiceLoss


def dice_loss_with_square(output, target, eps=1e-7) -> torch.Tensor:
    """
    Dice Loss with squares at denominator

    Args:
        output: model prediction, shape NCHW
        target: ground truth, shape NCHW
        eps: small value to avoid dividing by zero

    Returns:
        Dice loss with squares in the formula
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    output_stable_sigmoid = nn.LogSigmoid()(output).exp()
    intersection_mul = torch.clamp_min(
        torch.sum(output_stable_sigmoid * target),
        eps,
    )
    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network"
    union_squared = torch.clamp_min(
        torch.sum(torch.square(output_stable_sigmoid))
        + torch.sum(torch.square(target)),
        eps,
    )
    logit = 2 * intersection_mul / union_squared
    loss_batch = -torch.log(logit)
    return loss_batch


def init_loss(config):
    if config.loss_function == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss_function == "log_dice_squared":
        criterion = dice_loss_with_square
    elif config.loss_function == "dice":
        criterion = DiceLoss(mode="binary", log_loss=False, from_logits=True)
    elif config.loss_function == "log_dice":
        criterion = DiceLoss(mode="binary", log_loss=True, from_logits=True)
    else:
        raise Exception("Loss function not set, please check the config")
    return criterion
