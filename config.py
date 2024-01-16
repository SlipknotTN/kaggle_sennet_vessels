import configparser
import json


class ConfigParams(object):
    def __init__(self, file):
        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Train
        self.train_batch_size = config.getint("TRAIN", "batch_size")
        self.train_resize_before_crop = config.getint(
            "TRAIN", "train_resize_before_crop"
        )
        self.epochs = config.getint("TRAIN", "epochs")
        self.loss_function = config.get("TRAIN", "loss_function")
        self.optimizer = config.get("TRAIN", "optimizer")
        self.patience = config.getint("TRAIN", "patience", fallback=self.epochs)
        self.num_batches_train_loss_aggregation = config.getint(
            "TRAIN", "num_batches_train_loss_aggregation"
        )
        self.num_batches_preds_train_visualization_period = config.getint(
            "TRAIN", "num_batches_preds_train_visualization_period"
        )
        self.num_batches_preds_val_visualization_period = config.getint(
            "TRAIN", "num_batches_preds_val_visualization_period"
        )
        if self.optimizer == "SGD":
            self.learning_rate = config.getfloat("SGD", "learning_rate")
            self.momentum = config.getfloat("SGD", "momentum")
        elif self.optimizer == "ADAM":
            self.learning_rate = config.getfloat("ADAM", "learning_rate")
        else:
            raise Exception(f"Optimizer {self.optimizer} not supported")
        self.val_metrics_to_log = json.loads(config.get("TRAIN", "val_metrics_to_log"))
        self.val_metric_to_monitor = config.get("TRAIN", "val_metric_to_monitor")
        assert (
            self.val_metric_to_monitor in self.val_metrics_to_log
        ), f"val_metric_to_monitor {self.val_metric_to_monitor} not present in val_metrics_to_log"

        # Dataset
        self.train_dirs = json.loads(config.get("DATASET", "train_dirs"))
        self.val_dirs = json.loads(config.get("DATASET", "val_dirs"))
        self.test_dirs = json.loads(config.get("DATASET", "test_dirs"))

        # Model
        self.model_name = config.get("MODEL", "name")
        self.model_input_size = config.getint("MODEL", "input_size")
        self.model_input_channels = config.getint("MODEL", "input_channels", fallback=1)
        self.model_smp_model = config.get("MODEL", "smp_model", fallback=None)
        if self.model_smp_model:
            self.smp_encoder = config.get("MODEL", "smp_encoder")
            self.smp_encoder_weights = config.get("MODEL", "smp_encoder_weights", fallback=None)

        # Inference
        self.threshold = config.getfloat("INFERENCE", "threshold", fallback=None)
