import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepare dataset: save images as png and labels as numpy arrays",
    )
    parser.add_argument(
        "--dataset_path", required=True, type=str, help="Dataset root dir"
    )
    parser.add_argument(
        "--output_path", required=True, type=str, help="Output root dir"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Train
    train_dir = os.path.join(args.dataset_path, "train")

    for subset in tqdm(os.listdir(train_dir), desc="subset"):
        subset_images = (
            os.listdir(os.path.join(train_dir, subset, "images"))
            if os.path.exists(os.path.join(train_dir, subset, "images"))
            else []
        )
        subset_labels = os.listdir(os.path.join(train_dir, subset, "labels"))
        print(
            f"train {subset}: images {len(subset_images)}, labels {len(subset_labels)}"
        )
        if len(subset_images) > 0:
            for image_filepath in tqdm(subset_images, desc="image"):
                image = cv2.imread(
                    os.path.join(train_dir, subset, "images", image_filepath),
                    cv2.IMREAD_COLOR,
                )
                image_output_dir = os.path.join(
                    args.output_path, "train", subset, "images"
                )
                os.makedirs(image_output_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(image_output_dir, image_filepath[:-3] + "png"), image
                )
        if len(subset_labels) > 0:
            for label_filepath in tqdm(subset_labels, desc="label"):
                label = cv2.imread(
                    os.path.join(train_dir, subset, "labels", label_filepath),
                    cv2.IMREAD_GRAYSCALE,
                )
                label_png_output_dir = os.path.join(
                    args.output_path, "train", subset, "labels_png"
                )
                os.makedirs(label_png_output_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(label_png_output_dir, label_filepath[:-3] + "png"),
                    label,
                )
                label_npy_output_dir = os.path.join(
                    args.output_path, "train", subset, "labels_npy"
                )
                os.makedirs(label_npy_output_dir, exist_ok=True)
                label_gt_indexes = label == 255
                label_npy = np.zeros_like(label)
                label_npy[label_gt_indexes] = 1
                np.save(
                    os.path.join(label_npy_output_dir, label_filepath[:-3] + "npy"),
                    label_npy,
                )


if __name__ == "__main__":
    main()
