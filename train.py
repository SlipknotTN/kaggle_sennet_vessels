import argparse
import json
import os
import shutil
from collections import OrderedDict, defaultdict

import cv2
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.transforms import get_train_transform, get_val_transform
from loss import init_loss
from metrics.metrics import init_metrics
from model import init_model
from optimizer import init_optimizer
from utils import get_device, set_seed
from visualization.image import convert_to_image


def add_image_sample_to_tensorboard(
    writer: SummaryWriter,
    tag_prefix: str,
    global_step: int,
    image_sample: torch.Tensor,
    label_sample: torch.Tensor,
    pred_sample: torch.Tensor,
):
    # Create a single image, each input is already in the [0.0, 1.0] range
    # with CHW shape (also when grayscale)
    # TODO: Add label - prediction diff
    all_in_one_image = torch.concat(
        [image_sample.to("cpu"), label_sample.to("cpu"), pred_sample.to("cpu")], dim=-1
    )
    writer.add_image(
        f"{tag_prefix}/image_label_pred", all_in_one_image, global_step=global_step
    )


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train the model",
    )
    parser.add_argument(
        "--config_path", required=True, type=str, help="Configuration filepath"
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Dataset root dir"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--debug_augmentation",
        action="store_true",
        help="Visualize training augmented images instead of training",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    device = get_device()

    config = ConfigParams(args.config_path)
    print(f"Config: {json.dumps(config.__dict__, indent=4)}")
    set_seed(config.seed)

    model, preprocess_function, inverse_preprocess_function = init_model(config)
    model.to(device)
    total_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()
    print(f"Trainable parameters: {total_parameters}")
    print(f"Estimated size: {(total_parameters * 4 / 1024 / 1024):.2f} MB")
    print(model)

    # Init datasets and dataloaders
    data_transform_train = get_train_transform(config)
    data_transform_val = get_val_transform(config)
    train_dataset = BloodVesselDataset(
        [os.path.join(args.dataset_path, train_dir) for train_dir in config.train_dirs],
        data_transform_train,
        preprocess_function=preprocess_function,
        dataset_with_gt=True,
        in_channels=config.model_input_channels,
    )
    val_dataset = BloodVesselDataset(
        [os.path.join(args.dataset_path, val_dir) for val_dir in config.val_dirs],
        data_transform_val,
        preprocess_function=preprocess_function,
        dataset_with_gt=True,
        in_channels=config.model_input_channels,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, num_workers=4, shuffle=True
    )
    train_batches = len(train_dataset) // config.train_batch_size + int(
        len(train_dataset) % config.train_batch_size > 0
    )
    print(
        f"Train dataset samples: {len(train_dataset)}, "
        f"num_batches with batch size {config.train_batch_size}: {train_batches}"
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train_batch_size, num_workers=4, shuffle=False
    )
    val_batches = len(val_dataset) // config.train_batch_size + int(
        len(val_dataset) % config.train_batch_size > 0
    )
    print(
        f"Validation dataset samples: {len(val_dataset)}, "
        f"num_batches with batch size {config.train_batch_size}: {val_batches}"
    )

    # Train loop
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(logdir=args.output_dir)
    writer.add_graph(
        model.eval(),
        torch.zeros(
            [
                config.train_batch_size,
                config.model_input_channels,
                config.model_train_input_size,
                config.model_train_input_size,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    loss_criterion = init_loss(config)
    optimizer = init_optimizer(config, model)
    val_metrics = init_metrics(config)

    val_metric_to_monitor = None
    for val_metric in val_metrics:
        if val_metric.to_monitor:
            val_metric_to_monitor = val_metric
    assert val_metric_to_monitor is not None, "No val_metric_to_monitor"

    training_metrics = OrderedDict()

    best_monitored_metric_value = None
    best_epoch_1_index = 0
    consecutive_no_improvements = 0

    for epoch_id in tqdm(range(config.epochs), desc="epoch"):
        model.train(True)

        train_total_loss = 0.0
        running_loss = 0.0

        # Iterate on train batches and update weights using loss
        for batch_id, data in tqdm(
            enumerate(train_dataloader), desc="batch", total=train_batches
        ):
            global_step = epoch_id * train_batches + batch_id
            # get the input images and labels
            images = data["image"]
            labels = data["label_model_size"]

            if args.debug_augmentation:
                # FIXME: Support input image with more than one slice
                for image_idx in range(images.shape[0]):
                    image_denorm = inverse_preprocess_function(x_norm=images[image_idx])
                    original_file = data["file"][image_idx]
                    image_to_plot = convert_to_image(image_denorm)
                    original_image = cv2.imread(original_file)
                    cv2.imshow(
                        "image_transformed",
                        cv2.resize(
                            image_to_plot,
                            (image_to_plot.shape[1] // 2, image_to_plot.shape[0] // 2),
                        ),
                    )
                    cv2.imshow(
                        "image_original",
                        cv2.resize(
                            original_image,
                            (
                                original_image.shape[1] // 2,
                                original_image.shape[0] // 2,
                            ),
                        ),
                    )
                    cv2.waitKey(0)
                continue

            # Move to GPU
            labels_device = labels.to(device)
            images_device = images.to(device)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            preds_logits = model(images_device)
            preds_sigmoid = nn.Sigmoid()(preds_logits)

            # calculate the loss between predicted and output image
            loss = loss_criterion(preds_logits, labels_device)
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if (
                batch_id % config.num_batches_train_loss_aggregation
                == config.num_batches_train_loss_aggregation - 1
            ):
                avg_sample_loss = (
                    running_loss / config.num_batches_train_loss_aggregation
                )
                print(
                    "Epoch: {}, Batch: {}, train last {} batches avg. loss: {}".format(
                        epoch_id + 1,
                        batch_id + 1,
                        config.num_batches_train_loss_aggregation,
                        avg_sample_loss,
                    )
                )
                running_loss = 0.0
                writer.add_scalar(
                    f"train/loss_{config.num_batches_train_loss_aggregation}_batches_avg",
                    avg_sample_loss,
                    global_step=global_step,
                )

            train_total_loss += loss.item()

            # write predictions to tensorboard during training
            if (
                batch_id % config.num_batches_preds_train_visualization_period
                == config.num_batches_preds_train_visualization_period - 1
            ):
                # Draw images restoring the original image values range
                add_image_sample_to_tensorboard(
                    writer,
                    "train",
                    global_step,
                    # Select the first slice and readd the channel dimension
                    inverse_preprocess_function(torch.unsqueeze(images[0][0], dim=0)),
                    labels[0],
                    preds_sigmoid[0],
                )

        train_loss = train_total_loss / train_batches
        print("Epoch: {}, Train avg. loss: {}".format(epoch_id + 1, train_loss))
        writer.add_scalar(
            "train/loss_epoch_avg", train_loss, global_step=(epoch_id + 1)
        )
        training_metrics[epoch_id + 1] = {"train_loss": train_loss}

        # Iterate on validation batches
        # The eval model introduces a different w.r.t. training loss which can be significant
        # while the training is not yet stable
        model.eval()
        print(f"Epoch: {epoch_id + 1}, calculating validation metrics...")
        with torch.no_grad():
            val_total_metrics = defaultdict(float)
            for batch_id, data in tqdm(enumerate(val_dataloader), total=val_batches):
                global_step = epoch_id * train_batches + batch_id
                # get the input images and labels
                images = data["image"]
                labels = data["label_model_size"]

                # Move to GPU
                labels_device = labels.to(device)
                images_device = images.to(device)

                # forward pass to get outputs
                preds_logits = model(images_device)
                preds_sigmoid = nn.Sigmoid()(preds_logits)

                # calculate the metrics on validation batch
                for single_metric in val_metrics:
                    metric_value = single_metric.evaluate(preds_sigmoid, labels_device)
                    val_total_metrics[single_metric.name] += metric_value

                # TODO: Calculate 3D surface dice metric (target of the competition),
                # but this works only for single kidneys

                # write predictions to tensorboard during validation
                if (
                    batch_id % config.num_batches_preds_val_visualization_period
                    == config.num_batches_preds_val_visualization_period - 1
                ):
                    add_image_sample_to_tensorboard(
                        writer,
                        "val",
                        global_step,
                        # Select the first slice and readd the channel dimension
                        inverse_preprocess_function(
                            torch.unsqueeze(images[0][0], dim=0)
                        ),
                        labels[0],
                        preds_sigmoid[0],
                    )

            early_stop = False
            for single_metric in val_metrics:
                single_metric_name = single_metric.name
                single_metric_avg = val_total_metrics[single_metric_name] / val_batches
                print(
                    f"Epoch: {epoch_id + 1}, Validation avg. {single_metric_name}: {single_metric_avg}"
                )
                writer.add_scalar(
                    f"val/{single_metric_name}_avg",
                    single_metric_avg,
                    global_step=(epoch_id + 1),
                )
                training_metrics[epoch_id + 1][
                    f"val_{single_metric_name}"
                ] = single_metric_avg
                if single_metric.to_monitor:
                    monitored_metric_value = single_metric_avg
                    if single_metric.is_improved(
                        new_value=monitored_metric_value,
                        old_value=best_monitored_metric_value,
                    ):
                        print(
                            f"Epoch: {epoch_id + 1}, "
                            f"validation avg. {single_metric_name} improvement from {best_monitored_metric_value} "
                            f"to {monitored_metric_value}"
                        )
                        best_monitored_metric_value = monitored_metric_value
                        best_epoch_1_index = epoch_id + 1
                        output_model_filename = (
                            f"{args.output_dir}/{config.model_name}_{epoch_id + 1}.pt"
                        )
                        torch.save(model.state_dict(), output_model_filename)
                        print(f"Model saved to {output_model_filename}")
                        consecutive_no_improvements = 0
                    else:
                        print(
                            f"Epoch: {epoch_id + 1}, "
                            f"NO validation avg. {single_metric_name} improvement from {best_monitored_metric_value} "
                            f"to {monitored_metric_value}"
                        )
                        consecutive_no_improvements += 1
                        if consecutive_no_improvements > config.patience:
                            print(
                                f"Early stop, patience: {config.patience}, "
                                f"consecutive no improvements: {consecutive_no_improvements}"
                            )
                            early_stop = True

        writer.flush()

        if early_stop:
            break

    print(
        f"Train completed, best epoch {best_epoch_1_index} "
        f"with val avg. {val_metric_to_monitor.name} {best_monitored_metric_value}"
    )
    print(f"List of losses for each epoch: {training_metrics}")
    shutil.copy(args.config_path, os.path.join(args.output_dir, "config.cfg"))
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as out_fp:
        json.dump(training_metrics, out_fp, indent=4)

    # Copy transforms.py to keep track of the augmentations
    shutil.copy("./data/transforms.py", os.path.join(args.output_dir, "transforms.py"))

    writer.close()


if __name__ == "__main__":
    main()
