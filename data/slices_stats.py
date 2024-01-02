import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Measure dataset slices basic image stats",
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Dataset root dir"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Train
    train_dir = os.path.join(args.dataset_path, "train")

    slice_stats = {"mean_of_means": dict(), "mean_of_std_devs": dict()}
    for slice in tqdm(os.listdir(train_dir), desc="slice"):
        means = []
        std_devs = []
        slice_images = (
            os.listdir(os.path.join(train_dir, slice, "images"))
            if os.path.exists(os.path.join(train_dir, slice, "images"))
            else []
        )
        slice_labels = os.listdir(os.path.join(train_dir, slice, "labels"))
        print(f"train {slice}: images {len(slice_images)}, labels {len(slice_labels)}")
        for image_path in tqdm(slice_images, desc="slice_image"):
            image = cv2.imread(
                os.path.join(train_dir, slice, "images", image_path),
                cv2.IMREAD_GRAYSCALE,
            )
            means.append(np.mean(image))
            std_devs.append(np.std(image))
        slice_stats["mean_of_means"][slice] = np.mean(means)
        slice_stats["mean_of_std_devs"][slice] = np.mean(std_devs)
        slice_stats["num_images"][slice] = len(slice_images)
        print(
            f"Slice {slice}: mean_of_means {slice_stats['mean_of_means'][slice]}, "
            f"mean_of_std_devs {slice_stats['mean_of_std_devs'][slice]}"
        )

    print(json.dumps(slice_stats, indent=4))


if __name__ == "__main__":
    main()
