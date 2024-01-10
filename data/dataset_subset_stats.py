import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Measure dataset subsets basic image stats",
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

    subset_stats = {"mean_of_means": dict(), "mean_of_std_devs": dict()}
    for subset in tqdm(os.listdir(train_dir), desc="subset"):
        means = []
        std_devs = []
        subset_images = (
            os.listdir(os.path.join(train_dir, subset, "images"))
            if os.path.exists(os.path.join(train_dir, subset, "images"))
            else []
        )
        subset_labels = os.listdir(os.path.join(train_dir, subset, "labels"))
        print(f"train {subset}: images {len(subset_images)}, labels {len(subset_labels)}")
        for image_path in tqdm(subset_images, desc="subset_image"):
            image = cv2.imread(
                os.path.join(train_dir, subset, "images", image_path),
                cv2.IMREAD_GRAYSCALE,
            )
            means.append(np.mean(image))
            std_devs.append(np.std(image))
        subset_stats["mean_of_means"][subset] = np.mean(means)
        subset_stats["mean_of_std_devs"][subset] = np.mean(std_devs)
        subset_stats["num_images"][subset] = len(subset_images)
        print(
            f"Subset {subset}: mean_of_means {subset_stats['mean_of_means'][subset]}, "
            f"mean_of_std_devs {subset_stats['mean_of_std_devs'][subset]}"
        )

    print(json.dumps(subset_stats, indent=4))


if __name__ == "__main__":
    main()
