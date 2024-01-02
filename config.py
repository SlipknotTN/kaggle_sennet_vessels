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

        # Dataset
        self.train_dirs = json.loads(config.get("DATASET", "train_dirs"))
        self.val_dirs = json.loads(config.get("DATASET", "val_dirs"))
        self.test_dirs = json.loads(config.get("DATASET", "test_dirs"))

        # Model
        self.model_name = config.get("MODEL", "name")
        self.model_input_size = config.getint("MODEL", "input_size")
