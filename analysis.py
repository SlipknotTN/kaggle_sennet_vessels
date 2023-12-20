import argparse
import csv
import os
from collections import defaultdict


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Analyze dataset")
    parser.add_argument("--dataset_path", required=True, type=str, help="Dataset root dir")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Train
    train_dir = os.path.join(args.dataset_path, "train")

    print("File system")
    for subset in os.listdir(train_dir):
        subset_images = os.listdir(os.path.join(train_dir, subset, "images")) if os.path.exists(os.path.join(train_dir, subset, "images")) else []
        subset_labels = os.listdir(os.path.join(train_dir, subset, "labels"))
        print(f"train {subset}: images {len(subset_images)}, labels {len(subset_labels)}")

    csv_data_dict = defaultdict(list)
    with open(os.path.join(args.dataset_path, "train_rles.csv"), "r") as train_fp:
        reader = csv.reader(train_fp, delimiter=",")
        next(reader)
        for row in reader:
            full_data_id, rle = row[0], row[1]
            full_data_id_parts = full_data_id.split("_")
            subset_name = "_".join(full_data_id_parts[:-1])
            data_subset_id = int(full_data_id_parts[-1])
            # TODO: decode rle
            csv_data_dict[subset_name].append(data_subset_id)

    print("CSV file")
    for data_subset, data_rows in csv_data_dict.items():
        print(f"train subset {data_subset}: {len(data_rows)}")


if __name__ == "__main__":
    main()
