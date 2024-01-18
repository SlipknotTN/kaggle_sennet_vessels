import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.dice import DiceLoss

# TODO: Refactor this into a class and pass from_logits, log_loss and squared at denominator


def dice_loss(output, target, eps=1e-7, from_logits=True) -> torch.Tensor:
    """
    Dice Loss without log

    Args:
        output: model prediction, shape NCHW
        target: ground truth, shape NCHW, range [0, 1]
        eps: small value to avoid dividing by zero
        from_logits: True if input is logit, False if already in range [0, 1]

    Returns:
        Dice loss
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    if from_logits:
        output_stable = nn.LogSigmoid()(output).exp()
    else:
        output_stable = output
    intersection_mul = torch.sum(output_stable * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union = torch.sum(output_stable + target)
    union_clamped = torch.clamp_min(union, eps)
    # In case of GT and prediction empty, dice score is 1.0
    if union == 0.0:
        dice_score = torch.Tensor([1.0]).to(output.device)
    else:
        dice_score = 2 * intersection_mul / union_clamped
    return 1 - dice_score


def dice_log_loss(output, target, eps=1e-7, from_logits=True) -> torch.Tensor:
    """
    Dice Loss with log

    Args:
        output: model prediction, shape NCHW
        target: ground truth, shape NCHW, range [0, 1]
        eps: small value to avoid dividing by zero
        from_logits: True if input is logit, False if already in range [0, 1]

    Returns:
        Dice loss
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    if from_logits:
        output_stable = nn.LogSigmoid()(output).exp()
    else:
        output_stable = output
    intersection_mul = torch.sum(output_stable * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union = torch.sum(output_stable + target)
    union_clamped = torch.clamp_min(union, eps)
    # In case of GT and prediction empty, dice score is 1.0
    if union == 0.0:
        dice_score = 1.0
    else:
        dice_score = 2 * intersection_mul / union_clamped
    loss_batch = -torch.log(dice_score)
    return loss_batch


def dice_log_loss_with_square(
    output, target, eps=1e-7, from_logits=True
) -> torch.Tensor:
    """
    Dice Loss with log and squares at denominator

    Args:
        output: model prediction, shape NCHW
        target: ground truth, shape NCHW, range [0, 1]
        eps: small value to avoid dividing by zero
        from_logits: True if input is logit, False if already in range [0, 1]

    Returns:
        Dice loss with squares in the formula
    """
    # Comment from smp DiceLoss:
    # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
    # extreme values 0 and 1
    # The output values are not exactly the same
    if from_logits:
        output_stable = nn.LogSigmoid()(output).exp()
    else:
        output_stable = output
    intersection_mul = torch.sum(output_stable * target)

    # The difference w.r.t. smp DiceLoss is the additional square as mentioned in
    # "Fundus Images using Modified U-net Convolutional Neural Network" and the fact
    # that we don't consider if the ground truth is zero (the loss is zero in that case in smp dice loss)
    union_squared = torch.sum(torch.square(output_stable) + torch.square(target))
    union_clamped = torch.clamp_min(
        union_squared,
        eps,
    )
    # In case of GT and prediction empty, dice score is 1.0
    if union_squared == 0.0:
        dice_score = 1.0
    else:
        dice_score = 2 * intersection_mul / union_clamped
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
