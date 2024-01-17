import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.dice import DiceLoss


# TODO: Refactor this into a class

def dice_loss(output, target, eps=1e-7) -> torch.Tensor:
    """
    Dice Loss without log

    Args:
        output: model prediction (logit), shape NCHW
        target: ground truth, shape NCHW
        eps: small value to avoid dividing by zero

    Returns:
        Dice loss
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    output_stable_sigmoid = nn.LogSigmoid()(output).exp()
    intersection_mul = torch.sum(output_stable_sigmoid * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union_squared = torch.clamp_min(
        torch.sum(output_stable_sigmoid + target),
        eps,
    )
    dice_score = 2 * intersection_mul / union_squared
    return 1 - dice_score


def dice_log_loss(output, target, eps=1e-7) -> torch.Tensor:
    """
    Dice Loss with log

    Args:
        output: model prediction (logit), shape NCHW
        target: ground truth, shape NCHW
        eps: small value to avoid dividing by zero

    Returns:
        Dice loss
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    output_stable_sigmoid = nn.LogSigmoid()(output).exp()
    intersection_mul = torch.sum(output_stable_sigmoid * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union_squared = torch.clamp_min(
        torch.sum(output_stable_sigmoid + target),
        eps,
    )
    dice_score = 2 * intersection_mul / union_squared
    loss_batch = -torch.log(dice_score)
    return loss_batch


def dice_log_loss_with_square(output, target, eps=1e-7) -> torch.Tensor:
    """
    Dice Loss with log and squares at denominator

    Args:
        output: model prediction (logit), shape NCHW
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
    intersection_mul = torch.sum(output_stable_sigmoid * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union_squared = torch.clamp_min(
        torch.sum(torch.square(output_stable_sigmoid) + torch.square(target)),
        eps,
    )
    dice_score = 2 * intersection_mul / union_squared
    loss_batch = -torch.log(dice_score)
    return loss_batch


def init_loss(config):
    if config.loss_function == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss_function == "dice_loss":
        criterion = dice_loss
    elif config.loss_function == "log_dice_loss":
        criterion = dice_log_loss
    elif config.loss_function == "log_dice_loss_squared":
        criterion = dice_log_loss_with_square
    elif config.loss_function == "smp_dice_loss":
        criterion = DiceLoss(mode="binary", log_loss=False, from_logits=True)
    elif config.loss_function == "smp_log_dice_loss":
        criterion = DiceLoss(mode="binary", log_loss=True, from_logits=True)
    else:
        raise Exception("Loss function not set, please check the config")
    return criterion
