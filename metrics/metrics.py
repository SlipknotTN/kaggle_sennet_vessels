from abc import ABC, abstractmethod
from typing import List, Optional

from segmentation_models_pytorch.losses.dice import DiceLoss

from config import ConfigParams


class Metric(ABC):
    def __init__(self, to_monitor: bool):
        self._name = None
        self._to_monitor = to_monitor

    @abstractmethod
    def evaluate(self, output, target) -> float:
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
    1 - dice_loss (without log)
    """

    def __init__(self, to_monitor: bool):
        super().__init__(to_monitor)
        self._name = "dice_score"
        self.dice_loss_function = DiceLoss(
            mode="binary",
            log_loss=False,
            from_logits=False,
        )

    def evaluate(self, output, target) -> float:
        return 1 - self.dice_loss_function(output, target).item()

    def is_improved(self, new_value, old_value: Optional) -> bool:
        return new_value > old_value if old_value is not None else True


class DiceLossMetric(Metric):
    """
    Same of dice_loss (without log)
    """

    def __init__(self, to_monitor: bool):
        super().__init__(to_monitor)
        self._name = "dice_loss"
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
    metrics = []
    for val_metric in config.val_metrics_to_log:
        if val_metric == "dice_score":
            if val_metric == config.val_metric_to_monitor:
                metrics.append(DiceScore(to_monitor=True))
            else:
                metrics.append(DiceScore(to_monitor=False))
        elif val_metric == "dice_loss":
            if val_metric == config.val_metric_to_monitor:
                metrics.append(DiceLossMetric(to_monitor=True))
            else:
                metrics.append(DiceLossMetric(to_monitor=False))
        else:
            raise Exception(f"Metric {val_metric} unknown")
    return metrics
