"""
Visualize a pcd file with open3d library. You can create a pcd with the other visualization scripts
which start from numpy format.

Using jupyter notebook is not super smooth, although there is a tutorial https://www.open3d.org/html/tutorial/visualization/web_visualizer.html
I am having this issue setting webrtc server https://github.com/isl-org/Open3D/issues/6631

The exact code of this script is runnable from a jupyter notebook but the visualization
will be in the standalone visualizer, not inside the cells.

Example:
    python visualization/visualize_pcd.py \
    --input_pcd_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_pred_vs_label_rescaled_0.1_xyz.pcd
"""
import argparse
import os

import open3d as o3d


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize 3D point cloud file",
    )
    parser.add_argument(
        "--input_pcd_file",
        required=True,
        type=str,
        help="Numpy array 3D shaped filepath",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    assert os.path.exists(args.input_pcd_file), f"{args.input_pcd_file} does not exist"
    pcd = o3d.t.io.read_point_cloud(args.input_pcd_file)
    o3d.visualization.draw_geometries([pcd.to_legacy()])


if __name__ == "__main__":
    main()
