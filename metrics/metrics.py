from abc import ABC, abstractmethod
from typing import List, Optional

from segmentation_models_pytorch.losses.dice import DiceLoss

from config import ConfigParams
from loss import dice_loss


class Metric(ABC):
    def __init__(self, to_monitor: bool):
        self._name = None
        self._to_monitor = to_monitor

    @abstractmethod
    def evaluate(self, output, target) -> float:
        """
        Evaluate the metric on the model prediction and the target

        Args:
            output: output in range [0, 1] (sigmoid must be already applied)
            target: ground truth in the range [0, 1]

        Returns:
            Metric score
        """
        raise NotImplementedError

    @abstractmethod
    def is_improved(self, new_value, old_value: Optional) -> bool:
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def to_monitor(self):
        return self._to_monitor


class DiceScore(Metric):
    """
    DiceScore based on my custom dice loss class

    1 - dice_loss (without log)

    The diffence with the smp score is that the score is 0
    when the GT is 0 and the prediction no (instead of 1)
    """

    def __init__(self, to_monitor: bool):
        super().__init__(to_monitor)
        self._name = "dice_score"
        self.dice_loss_function = dice_loss

    def evaluate(self, output, target) -> float:
        return 1 - self.dice_loss_function(output, target, from_logits=False).item()

    def is_improved(self, new_value, old_value: Optional) -> bool:
        return new_value > old_value if old_value is not None else True


class SMPDiceScore(Metric):
    """
    Segmentation Models PyTorch dice score based on DiceLoss class

    1 - dice_loss (without log)

    WARNING: dice_loss is 0 when the GT is 0 even with FP predictions
    """

    def __init__(self, to_monitor: bool):
        super().__init__(to_monitor)
        self._name = "smp_dice_score"
        self.dice_loss_function = DiceLoss(
            mode="binary",
            log_loss=False,
            from_logits=False,
        )

    def evaluate(self, output, target) -> float:
        return 1 - self.dice_loss_function(output, target).item()

    def is_improved(self, new_value, old_value: Optional) -> bool:
        return new_value > old_value if old_value is not None else True


class SMPDiceLossMetric(Metric):
    """
    Segmentation Models PyTorch DiceLoss (without log)

    WARNING: dice_loss is 0 when the GT is 0 even with FP predictions
    """

    def __init__(self, to_monitor: bool):
        super().__init__(to_monitor)
        self._name = "smp_dice_loss"
        self.dice_loss_function = DiceLoss(
            mode="binary",
            log_loss=False,
            from_logits=False,
        )

    def evaluate(self, output, target) -> float:
        return self.dice_loss_function(output, target).item()

    def is_improved(self, new_value, old_value: Optional) -> bool:
        return new_value < old_value if old_value is not None else True


def init_metrics(config: ConfigParams) -> List[Metric]:
    """
    Initialize metrics classe

    Args:
        config: configuration parameters

    Returns:
        List of metrics to calculate
    """
    assert (
        config.val_metric_to_monitor in config.val_metrics_to_log
    ), f"config.val_metrics_monitored {config.val_metric_to_monitor} not present in config.val_metrics_logged"
    metrics_list = []
    for val_metric_name in config.val_metrics_to_log:
        if val_metric_name == "dice_score":
            if val_metric_name == config.val_metric_to_monitor:
                metrics_list.append(DiceScore(to_monitor=True))
            else:
                metrics_list.append(SMPDiceScore(to_monitor=False))
        elif val_metric_name == "smp_dice_score":
            if val_metric_name == config.val_metric_to_monitor:
                metrics_list.append(SMPDiceScore(to_monitor=True))
            else:
                metrics_list.append(SMPDiceScore(to_monitor=False))
        elif val_metric_name == "smp_dice_loss":
            if val_metric_name == config.val_metric_to_monitor:
                metrics_list.append(SMPDiceLossMetric(to_monitor=True))
            else:
                metrics_list.append(SMPDiceLossMetric(to_monitor=False))
        else:
            raise Exception(f"Metric {val_metric_name} unknown")
    return metrics_list
