import argparse
import glob
import json

import pandas as pd


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Parse metric results from json files and create a CSV file",
    )
    parser.add_argument("--input_dir", required=True, type=str, help="Root input dir")
    parser.add_argument(
        "--ignore_dirs",
        required=False,
        default=[],
        nargs="*",
        type=str,
        help="Ignore subdirectories",
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="Destination image width"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    metric_names = ["avg_2D_dice_score", "surface_dice_score"]

    # First subdir: category (train-test dataset or submissions)
    # Second subdir: model
    # Third subdir: eval run
    # File: metrics.json containing dataset_name and metrics

    metric_filepaths = sorted(glob.glob(args.input_dir + "/*/*/*/metrics.json"))

    df = pd.DataFrame(
        columns=[
            "category",
            "eval_dataset",
            "model_name",
            "eval_run",
            "avg_2D_dice_score",
            "surface_dice_score",
        ]
    )

    for metric_filepath in metric_filepaths:
        metrics_dict = json.load(open(metric_filepath))
        category = metric_filepath.split("/")[-4]
        model_name = metric_filepath.split("/")[-3]
        eval_run = metric_filepath.split("/")[-2]
        for dataset, dataset_metrics in metrics_dict.items():
            avg_2D_dice_score = f"{float(dataset_metrics['avg_2D_dice_score']):.3f}"
            surface_dice_score = f"{float(dataset_metrics['surface_dice_score']):.3f}"
            df_row = pd.DataFrame.from_dict(
                {
                    "eval_dataset": [dataset],
                    "category": [category],
                    "model_name": [model_name],
                    "eval_run": [eval_run],
                    "avg_2D_dice_score": [avg_2D_dice_score],
                    "surface_dice_score": [surface_dice_score],
                }
            )
            df = pd.concat([df, df_row])

    df.sort_values(
        by=["category", "eval_dataset", "surface_dice_score"],
        ascending=[True, True, False],
        inplace=True,
    )

    df.to_csv(args.output_file, index=False, sep=";")
    print(f"Scores saved to {args.output_file}")


if __name__ == "__main__":
    main()
