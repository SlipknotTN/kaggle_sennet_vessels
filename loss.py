import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.dice import DiceLoss


def iou_loss(output, target):
    batch_size = output.shape[0]
    loss_components = []
    for batch_id in range(batch_size):
        # Use minimum values to avoid NaN, target is simply HW,
        # so we need to get rid of the channel dimension of the image
        intersection_mul = torch.max(
            torch.sum(output[batch_id][0] * target[batch_id]),
            torch.Tensor([0.001]).to(output.device),
        )
        union_squared = torch.max(
            torch.sum(torch.square(output[batch_id][0]))
            + torch.sum(torch.square(target[batch_id])),
            torch.Tensor([0.001]).to(output.device),
        )
        logit = 2 * intersection_mul / union_squared
        loss_component = -torch.log(logit)
        loss_components.append(loss_component)
    return torch.mean(torch.stack(loss_components))


def init_loss(config):
    if config.loss_function == "BCE":
        criterion = nn.BCEWithLogitsLoss()
        input_logit = True
    elif config.loss_function == "IOU":
        criterion = iou_loss
        input_logit = False
    elif config.loss_function == "dice":
        criterion = DiceLoss(mode="binary", log_loss=False, from_logits=True)
        input_logit = True
    elif config.loss_function == "log_dice":
        criterion = DiceLoss(mode="binary", log_loss=True, from_logits=True)
        input_logit = True
    else:
        raise Exception("Loss function not set, please check the config")
    return criterion, input_logit
