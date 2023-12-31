import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ConfigParams
from data.dataset import BloodVesselDataset
from data.transforms import get_train_transform, get_val_transform
from model import UnetModel
from utils import get_device


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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    training_metrics = OrderedDict()

    best_val_loss = sys.float_info.max
    best_epoch_1_index = 0

    for epoch_id in tqdm(range(config.epochs), desc="epoch"):
        model.train()

        train_total_loss = 0.0
        running_loss = 0.0

        # Iterate on train batches and update weights using loss
        for batch_id, data in tqdm(
            enumerate(train_dataloader), desc="batch", total=train_batches
        ):
            # get the input images and labels
            images = data["image"]
            labels = data["label"]
            # print(f"Images shape: {images.shape}, labels shape: {labels.shape}")

            # Move to GPU
            labels = labels.to(device)
            images = images.to(device)

            # forward pass to get outputs
            output = model(images)

            # calculate the loss between predicted and output image
            loss = criterion(output, labels)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_id % 10 == 9:  # print every 10 batches
                print(
                    "Epoch: {}, Batch: {}, train last 10 batches avg. sample loss: {}".format(
                        epoch_id + 1,
                        batch_id + 1,
                        running_loss / 10 / config.train_batch_size,
                    )
                )
                running_loss = 0.0

            train_total_loss += loss.item()

        train_loss = train_total_loss / train_batches / config.train_batch_size
        print("Epoch: {}, Train avg. sample loss: {}".format(epoch_id + 1, train_loss))

        # Iterate on validation batches
        print(f"Epoch: {epoch_id + 1}, calculating validation loss...")
        with torch.no_grad():
            model.eval()
            val_total_loss = 0.0
            for batch_i, data in tqdm(enumerate(val_dataloader), total=val_batches):
                # get the input images and labels
                images = data["image"]
                labels = data["label"]

                # Move to GPU
                labels = labels.to(device)
                images = images.to(device)

                # forward pass to get outputs
                output = model(images)

                # calculate the loss between predicted and output image
                loss = criterion(output, labels)

                val_total_loss += loss.item()

                # TODO: Calculate dice metric

        # TODO: Visualize train and val predictions during training

        val_loss = val_total_loss / val_batches / config.train_batch_size
        print(
            "Epoch: {}, Validation avg. sample loss: {}".format(epoch_id + 1, val_loss)
        )
        if val_loss < best_val_loss:
            print(
                f"Epoch: {epoch_id + 1}, validation loss improvement from {best_val_loss} to {val_loss}"
            )
            best_val_loss = val_loss
            best_epoch_1_index = epoch_id + 1
            os.makedirs(args.output_dir, exist_ok=True)
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

    print(
        f"Train completed, best epoch {best_epoch_1_index} with val loss {best_val_loss}"
    )
    print(f"List of losses for each epoch: {training_metrics}")
    shutil.copy(args.config_path, os.path.join(args.output_dir, "config.cfg"))
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as out_fp:
        json.dump(training_metrics, out_fp, indent=4)


if __name__ == "__main__":
    main()
