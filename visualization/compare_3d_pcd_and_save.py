"""
Compare two 3D point clouds stored as PCD format:
- Load reference and other point clouds (intensities metadata is necessary, use "visualize_3d_npy_and_save_pcd.py to
create a compatible file)
- Build a new point cloud highlighting true positives, false positives and false negatives using the
first volume as reference (typically the ground truth)
- [Optional] Save the new point cloud as PCD format

This script does not apply any quantization to the original point clouds

Coordinates system: xyz right-handed, z-up.
Slices (z) going from bottom (0) to up (max = num_slices - 1)

Example:
    python visualization/compare_3d_pcd.py \
    --reference_file /media/michele/DATA-2/Datasets/kaggle_vessels/3D/train/kidney_3_sparse/kidney_3_sparse_label_xyz.pcd \
    --compare_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_prediction_xyz.pcd \
    --output_pcd_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_label_vs_pred_xyz.pcd
"""
import argparse
import os

import numpy as np
import open3d as o3d


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two 3D pcd point clouds using one as reference for the other",
    )
    parser.add_argument(
        "--reference_file",
        required=True,
        type=str,
        help="Reference PCD filepath, typically the ground truth, but it can be any",
    )
    parser.add_argument(
        "--compare_file",
        required=True,
        type=str,
        help="PCD to compare filepath",
    )
    parser.add_argument(
        "--output_pcd_file", required=False, type=str, help="Output pcd filepath"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    # Please notice that all the Tensors are loaded into CPU

    print(f"Loading PCD reference from {args.reference_file}")
    assert os.path.exists(args.reference_file), f"{args.reference_file} does not exist"
    ref_pcd = o3d.t.io.read_point_cloud(args.reference_file)

    print(f"Loading PCD to compare from {args.compare_file}")
    assert os.path.exists(args.compare_file), f"{args.compare_file} does not exist"
    comp_pcd = o3d.t.io.read_point_cloud(args.compare_file)

    # Avoid using numpy array unless strictly necessary, getting numpy array is a slow operation
    # ref_points = np.asarray(ref_pcd.point.positions)
    # ref_confidences = np.asarray(ref_pcd.point.intensities)
    # comp_points = np.asarray(comp_pcd.point.positions)
    # comp_confidences = np.asarray(comp_pcd.point.intensities)

    print("Extracting TP and FN...")
    ref_comp_dists = ref_pcd.to_legacy().compute_point_cloud_distance(
        comp_pcd.to_legacy()
    )
    ref_comp_dists_npy = np.asarray(ref_comp_dists)
    ref_tp_indexes = np.where(ref_comp_dists_npy == 0.0)[0]
    ref_fn_indexes = np.where(ref_comp_dists_npy > 0.0)[0]

    print("Extracting FP...")
    comp_ref_dists = comp_pcd.to_legacy().compute_point_cloud_distance(
        ref_pcd.to_legacy()
    )
    comp_ref_dists_npy = np.asarray(comp_ref_dists)
    comp_fp_indexes = np.where(comp_ref_dists_npy > 0.0)[0]

    tp_xyz_coords = ref_pcd.select_by_index(ref_tp_indexes).point.positions
    print(f"True positives: {tp_xyz_coords.shape[0]}")
    fn_xyz_coords = ref_pcd.select_by_index(ref_fn_indexes).point.positions
    print(f"False negatives: {fn_xyz_coords.shape[0]}")
    fp_xyz_coords = comp_pcd.select_by_index(comp_fp_indexes).point.positions
    print(f"False positives: {fp_xyz_coords.shape[0]}")
    all_xyz_coords = tp_xyz_coords.append(fn_xyz_coords, axis=0).append(
        fp_xyz_coords, axis=0
    )
    assert (
        all_xyz_coords.shape[0]
        == tp_xyz_coords.shape[0] + fn_xyz_coords.shape[0] + fp_xyz_coords.shape[0]
    ), "Total TP + FN + FP not correct"

    # Set the RGB colors: the color Tensor indexes are matched with the position Tensor indexes
    # TP - Green
    tp_colors = o3d.core.Tensor.empty(
        shape=list(tp_xyz_coords.shape),
        dtype=o3d.core.float32,
        device=tp_xyz_coords.device,
    )
    tp_colors[:, :] = [0.0, 1.0, 0.0]
    # FN - Blue
    fn_colors = o3d.core.Tensor.empty(
        shape=list(fn_xyz_coords.shape),
        dtype=o3d.core.float32,
        device=fn_xyz_coords.device,
    )
    fn_colors[:, :] = [0.0, 0.0, 1.0]
    # FP - Red
    fp_colors = o3d.core.Tensor.empty(
        shape=list(fp_xyz_coords.shape),
        dtype=o3d.core.float32,
        device=fp_xyz_coords.device,
    )
    fp_colors[:, :] = [1.0, 0.0, 0.0]
    all_colors = tp_colors.append(fn_colors, axis=0).append(fp_colors, axis=0)
    assert (
        all_colors.shape[0]
        == tp_colors.shape[0] + fn_colors.shape[0] + fp_colors.shape[0]
    ), "Total colors not correct"

    # Save point cloud to PCD format
    print(f"Output points: {all_xyz_coords.shape[0]}")
    output_pcd = o3d.t.geometry.PointCloud()
    # Set the colors for the points
    output_pcd.point.positions = all_xyz_coords
    output_pcd.point.colors = all_colors
    o3d.visualization.draw_geometries([output_pcd.to_legacy()])
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        o3d.t.io.write_point_cloud(args.output_pcd_file, output_pcd)
        print(f"PCD file save to {args.output_pcd_file}")


if __name__ == "__main__":
    main()
