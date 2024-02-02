import argparse

import cv2
import os
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Resize images in a directory to a different size keeping the image format",
    )
    parser.add_argument(
        "--input_path", required=True, type=str, help="Input images dir"
    )
    parser.add_argument(
        "--output_path", required=True, type=str, help="Output images dir"
    )
    parser.add_argument(
        "--width", required=True, type=int, help="Destination image width"
    )
    parser.add_argument(
        "--height", required=True, type=int, help="Destination image height"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    os.makedirs(args.output_path, exist_ok=True)

    # Assuming all the files in the directory are images
    for image_filename in tqdm(sorted(os.listdir(args.input_path))):
        print(image_filename)
        image = cv2.imread(os.path.join(args.input_path, image_filename), cv2.IMREAD_COLOR)
        rsz_image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.output_path, image_filename), rsz_image)


if __name__ == "__main__":
    main()
