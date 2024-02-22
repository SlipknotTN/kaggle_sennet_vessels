import argparse
import csv
import os

import cv2
import numpy as np


def rle_encode(img):
    """
    ref.: https://www.kaggle.com/stainsby/fast-tested-rle + fix for all zero image

    Args:
        img: numpy array, 1 - mask, 0 - background

    Returns:
        run length as string formatted
    """
    pixels = img.flatten()
    if np.max(pixels) == np.min(pixels) == 0:
        return "1 0"
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height,width) of array to return
    Returns:
         numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Test decode and re-encode rle format",
    )
    parser.add_argument(
        "--input_root_dir",
        required=True,
        type=str,
        help="Directory including the train_rles.csv file",
    )
    args = parser.parse_args()
    return args


def main():
    # Test decode and re-encode correctness
    args = do_parsing()
    print(args)
    train_rles_filepath = os.path.join(args.input_root_dir, "train_rles.csv")
    num_rows = 0
    with open(train_rles_filepath, "r") as in_fp:
        reader = csv.reader(in_fp, delimiter=",")
        next(reader)
        for row in reader:
            full_data_id, rle = row[0], row[1]
            full_data_id_parts = full_data_id.split("_")
            subset_name = "_".join(full_data_id_parts[:-1])
            image_name = full_data_id_parts[-1]
            label_filepath = os.path.join(
                args.input_root_dir, "train", subset_name, "labels", f"{image_name}.tif"
            )
            print(f"Label filepath: {label_filepath}, rle {rle}")
            label = cv2.imread(label_filepath, cv2.IMREAD_GRAYSCALE)
            print(
                f"Raw label shape: {label.shape}, dtype {label.dtype} "
                + f"min {np.min(label)}, max {np.max(label)}"
            )
            decoded_rle = rle_decode(rle, (label.shape[0], label.shape[1]))
            print(
                f"Decoded rle shape: {decoded_rle.shape}, dtype {decoded_rle.dtype} "
                + f"min {np.min(decoded_rle)}, max {np.max(decoded_rle)}"
            )
            label_norm = (label / 255).astype(np.uint8)
            re_encoded_rle = rle_encode(label_norm)
            assert np.array_equal(decoded_rle, label_norm), f"decoded rle != image"
            assert (
                re_encoded_rle == rle
            ), f"re-encoded rle {re_encoded_rle} != original rle {rle}"
            num_rows += 1


if __name__ == "__main__":
    main()
