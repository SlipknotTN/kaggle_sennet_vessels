import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.transforms import get_train_transform, get_val_transform
from model import UnetModel
from utils import get_device


def add_image_sample_to_tensorboard(
    writer, tag_prefix, global_step, image_sample, label_sample, pred_sample
):
    writer.add_image(f"{tag_prefix}/image", image_sample, global_step=global_step)
    writer.add_image(f"{tag_prefix}/label", label_sample, global_step=global_step)
    writer.add_image(f"{tag_prefix}/pred", pred_sample, global_step=global_step)


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
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    device = get_device()

    config = ConfigParams(args.config_path)
    print(f"Config: {config.__dict__}")

    # Init model
    model = UnetModel()
    model.to(device)
    total_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()
    print(f"Trainable parameters: {total_parameters}")
    print(f"Estimated size: {(total_parameters * 4 / 1024 / 1024):.2f} MB")
    print(model)

    # Init datasets and dataloades
    data_transform_train = get_train_transform(config)
    data_transform_val = get_val_transform(config)
    train_dataset = BloodVesselDataset(
        [os.path.join(args.dataset_path, train_dir) for train_dir in config.train_dirs],
        data_transform_train,
        dataset_with_gt=True,
    )
    val_dataset = BloodVesselDataset(
        [os.path.join(args.dataset_path, val_dir) for val_dir in config.val_dirs],
        data_transform_val,
        dataset_with_gt=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )
    train_batches = len(train_dataset) // config.train_batch_size + int(
        len(train_dataset) % config.train_batch_size > 0
    )
    print(
        f"Train dataset samples: {len(train_dataset)}, num_batches with batch size {config.train_batch_size}: {train_batches}"
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train_batch_size, shuffle=False
    )
    val_batches = len(val_dataset) // config.train_batch_size + int(
        len(val_dataset) % config.train_batch_size > 0
    )
    print(
        f"Validation dataset samples: {len(val_dataset)}, num_batches with batch size {config.train_batch_size}: {val_batches}"
    )

    # Train loop
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(logdir=args.output_dir)
    writer.add_graph(
        model,
        torch.zeros(
            [
                config.train_batch_size,
                1,
                config.model_input_size,
                config.model_input_size,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    training_metrics = OrderedDict()

    best_val_loss = sys.float_info.max
    best_epoch_1_index = 0

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
            labels = data["label"]
            # print(f"Images shape: {images.shape}, labels shape: {labels.shape}")

            # Move to GPU
            labels_device = labels.to(device)
            images_device = images.to(device)

            # forward pass to get outputs
            preds = model(images_device)

            # calculate the loss between predicted and output image
            loss = criterion(preds, labels_device)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if (
                batch_id % config.num_batches_train_loss_aggregation
                == config.num_batches_train_loss_aggregation
            ):
                avg_sample_loss = running_loss / 10 / config.train_batch_size
                print(
                    "Epoch: {}, Batch: {}, train last 10 batches avg. sample loss: {}".format(
                        epoch_id + 1,
                        batch_id + 1,
                        avg_sample_loss,
                    )
                )
                running_loss = 0.0
                writer.add_scalar(
                    "train/loss_10_batches_avg",
                    avg_sample_loss,
                    global_step=global_step,
                )

            train_total_loss += loss.item()

            # write predictions to tensorboard during training
            if (
                batch_id % config.num_batches_preds_visualization_period
                == config.num_batches_preds_visualization_period - 1
            ):
                add_image_sample_to_tensorboard(
                    writer,
                    "train",
                    global_step,
                    images[0],
                    labels[0],
                    nn.Sigmoid()(preds[0]),
                )

        train_loss = train_total_loss / train_batches / config.train_batch_size
        print("Epoch: {}, Train avg. sample loss: {}".format(epoch_id + 1, train_loss))

        # Iterate on validation batches
        model.eval()
        print(f"Epoch: {epoch_id + 1}, calculating validation loss...")
        with torch.no_grad():
            val_total_loss = 0.0
            for batch_id, data in tqdm(enumerate(val_dataloader), total=val_batches):
                global_step = epoch_id * train_batches + batch_id
                # get the input images and labels
                images = data["image"]
                labels = data["label"]

                # Move to GPU
                labels_device = labels.to(device)
                images_device = images.to(device)

                # forward pass to get outputs
                preds = model(images_device)

                # calculate the loss between predicted and output image
                loss = criterion(preds, labels_device)

                val_total_loss += loss.item()

                # TODO: Calculate dice metric

                # write predictions to tensorboard during validation
                if (
                    batch_id % config.num_batches_preds_visualization_period
                    == config.num_batches_preds_visualization_period - 1
                ):
                    add_image_sample_to_tensorboard(
                        writer,
                        "val",
                        global_step,
                        images[0],
                        labels[0],
                        nn.Sigmoid()(preds[0]),
                    )

        val_loss = val_total_loss / val_batches / config.train_batch_size
        print(
            "Epoch: {}, Validation avg. sample loss: {}".format(epoch_id + 1, val_loss)
        )
        writer.add_scalar("val/loss_epoch_avg", val_loss, global_step=(epoch_id + 1))
        if val_loss < best_val_loss:
            print(
                f"Epoch: {epoch_id + 1}, validation loss improvement from {best_val_loss} to {val_loss}"
            )
            best_val_loss = val_loss
            best_epoch_1_index = epoch_id + 1
            output_model_filename = (
                f"{args.output_dir}/{config.model_name}_{epoch_id + 1}.pt"
            )
            torch.save(model.state_dict(), output_model_filename)
            print(f"Model saved to {output_model_filename}")
        else:
            print(
                f"Epoch: {epoch_id + 1}, NO validation loss improvement from {best_val_loss} to {val_loss}"
            )
            # TODO: Implement patience to stop earlier

        training_metrics[epoch_id + 1] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        writer.flush()

    print(
        f"Train completed, best epoch {best_epoch_1_index} with val loss {best_val_loss}"
    )
    print(f"List of losses for each epoch: {training_metrics}")
    shutil.copy(args.config_path, os.path.join(args.output_dir, "config.cfg"))
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as out_fp:
        json.dump(training_metrics, out_fp, indent=4)

    writer.close()


if __name__ == "__main__":
    main()
