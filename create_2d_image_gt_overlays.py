import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Create pairs: image and image with gt overlays")
    parser.add_argument("--dataset_path", required=True, type=str, help="Dataset root dir")
    parser.add_argument("--output_path", required=True, type=str, help="Output root dir")
    parser.add_argument("--debug", action="store_true", help="Visualize images with gt overlay")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Train
    train_dir = os.path.join(args.dataset_path, "train")

    for subset in tqdm(os.listdir(train_dir), desc="subset"):
        subset_images = os.listdir(os.path.join(train_dir, subset, "images")) if os.path.exists(os.path.join(train_dir, subset, "images")) else []
        subset_labels = os.listdir(os.path.join(train_dir, subset, "labels"))
        try:
            assert len(subset_images) == len(subset_labels), f"Number of images {len(subset_images)}!= number of labels {len(subset_labels)} for subset {subset}"
            print(f"train {subset}: images {len(subset_images)}, labels {len(subset_labels)}")
            if len(subset_images) > 0:
                output_dir = os.path.join(args.output_path, "train", subset)
                os.makedirs(output_dir, exist_ok=True)
                for image_filepath in tqdm(subset_images, desc="image"):
                    image = cv2.imread(os.path.join(train_dir, subset, "images", image_filepath), cv2.IMREAD_COLOR)
                    label = cv2.imread(os.path.join(train_dir, subset, "labels", image_filepath), cv2.IMREAD_GRAYSCALE)
                    # Convert label to red
                    label = np.expand_dims(label, axis=-1)
                    label_red = np.concatenate([np.zeros_like(label), np.zeros_like(label), label], axis=-1)
                    image_with_gt_overlay = cv2.addWeighted(image, 1, label_red, 0.5, 0)
                    image_pairs = np.concatenate([image, image_with_gt_overlay], axis=1)
                    if args.debug:
                        cv2.imshow("image_pairs", image_pairs)
                        cv2.waitKey(0)
                    cv2.imwrite(os.path.join(output_dir, image_filepath[:-3] + "png"), image_pairs)
        except AssertionError as e:
            print(f"{subset}: {e}")

if __name__ == "__main__":
    main()
