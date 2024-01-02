import argparse
import csv
import os
from collections import defaultdict


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Count dataset slices",
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

    print("File system")
    for slice in os.listdir(train_dir):
        slice_images = (
            os.listdir(os.path.join(train_dir, slice, "images"))
            if os.path.exists(os.path.join(train_dir, slice, "images"))
            else []
        )
        slice_labels = os.listdir(os.path.join(train_dir, slice, "labels"))
        print(f"train {slice}: images {len(slice_images)}, labels {len(slice_labels)}")

    csv_data_dict = defaultdict(list)
    with open(os.path.join(args.dataset_path, "train_rles.csv"), "r") as train_fp:
        reader = csv.reader(train_fp, delimiter=",")
        next(reader)
        for row in reader:
            full_data_id, rle = row[0], row[1]
            full_data_id_parts = full_data_id.split("_")
            slice_name = "_".join(full_data_id_parts[:-1])
            data_slice_id = int(full_data_id_parts[-1])
            csv_data_dict[slice_name].append(data_slice_id)

    print("CSV file")
    for data_slice, data_rows in csv_data_dict.items():
        print(f"train slice {data_slice}: {len(data_rows)}")


if __name__ == "__main__":
    main()
