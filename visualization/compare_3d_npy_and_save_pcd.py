"""
Compare two 3D point clouds initially stored as numpy arrays:
- Load reference and other volume as 3D numpy full resolution points cloud for a single kidney
- Rescale them to reduce the memory footprint
- Build a new rescaled point cloud highlighting true positives, false positives and false negatives using the
first volume as reference (typically the ground truth)
- [Optional] Save the new point cloud as PCD format

Coordinates system: xyz right-handed, z-up.
Slices (z) going from bottom (0) to up (max = num_slices - 1)

Example:
    python visualization/compare_3d_npy_and_save_pcd.py \
    --reference_file /media/michele/DATA-2/Datasets/kaggle_vessels/3D/train/kidney_3_sparse/kidney_3_sparse_label_xyz.npy \
    --compare_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_prediction_xyz.npy \
    --rescale_factor 0.1 \
    --output_pcd_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_label_vs_pred_rescaled_0.1_xyz.pcd
"""
import argparse

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two 3D point clouds using one as reference for the other",
    )
    parser.add_argument(
        "--reference_file",
        required=True,
        type=str,
        help="Reference numpy 3D array filepath, typically the ground truth, but it can be any",
    )
    parser.add_argument(
        "--compare_file",
        required=True,
        type=str,
        help="Numpy 3D array to compare filepath",
    )
    parser.add_argument(
        "--rescale_factor",
        required=False,
        default=1.0,
        type=float,
        help="Rescale factor to avoid OOM, e.g. 0.1",
    )
    parser.add_argument(
        "--output_pcd_file", required=False, type=str, help="Output pcd filepath"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device to load Open3D tensors",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    assert (
        0.0 <= args.rescale_factor <= 1.0
    ), f"Rescale factor must be in [0.0, 1.0] range"

    if args.device == "cpu":
        device = o3d.core.Device("CPU:0")
    elif args.device == "gpu":
        device = o3d.core.Device("CUDA:0")
    else:
        raise ValueError(f"Device not supported {args.device}")

    print(f"Loading reference volume from {args.reference_file}")
    ref_volume_xyz = np.load(args.reference_file)
    print(
        f"Loaded reference 3D shape width x height X num_slices (xyz): {ref_volume_xyz.shape}"
    )
    if args.rescale_factor < 1.0:
        print("Rescaling reference numpy volume")
        rescaled_ref_volume = zoom(
            ref_volume_xyz,
            (args.rescale_factor, args.rescale_factor, args.rescale_factor),
        )
        print(
            f"Rescaled reference 3D shape to width x height X num_slices (xyz): {rescaled_ref_volume.shape}"
        )
        # Save memory
        del ref_volume_xyz
    else:
        print("No rescaling")
        rescaled_ref_volume = ref_volume_xyz

    print(f"Loading volume to compare from {args.compare_file}")
    comp_volume_xyz = np.load(args.compare_file)
    print(
        f"Loaded 3D shape to compare width x height X num_slices (xyz): {comp_volume_xyz.shape}"
    )
    if args.rescale_factor < 1.0:
        print("Rescaling numpy volume to compare")
        rescaled_comp_volume = zoom(
            comp_volume_xyz,
            (args.rescale_factor, args.rescale_factor, args.rescale_factor),
        )
        print(
            f"Rescaled 3D shape to compare to width x height X num_slices (xyz): {rescaled_comp_volume.shape}"
        )
        del comp_volume_xyz
    else:
        rescaled_comp_volume = comp_volume_xyz

    assert rescaled_comp_volume.shape == rescaled_ref_volume.shape

    # Draw point cloud with green TP, blue FN, red FP
    tp_xyz_coords = np.argwhere(
        (rescaled_comp_volume > 0.0) & (rescaled_ref_volume > 0.0)
    )
    fn_xyz_coords = np.argwhere(
        (rescaled_comp_volume == 0.0) & (rescaled_ref_volume > 0.0)
    )
    fp_xyz_coords = np.argwhere(
        (rescaled_comp_volume > 0.0) & (rescaled_ref_volume == 0.0)
    )
    tp_xyz_colors = np.zeros_like(tp_xyz_coords)
    tp_xyz_colors[:, :] = [0.0, 1.0, 0.0]
    fn_xyz_colors = np.zeros_like(fn_xyz_coords)
    fn_xyz_colors[:, :] = [0.0, 0.0, 1.0]
    fp_xyz_colors = np.zeros_like(fp_xyz_coords)
    fp_xyz_colors[:, :] = [1.0, 0.0, 0.0]

    xyz_coords = np.concatenate([tp_xyz_coords, fn_xyz_coords, fp_xyz_coords], axis=0)
    xyz_colors = np.concatenate([tp_xyz_colors, fn_xyz_colors, fp_xyz_colors], axis=0)
    # Save point cloud to PCD format, o3d visualizer is way faster than matplotlib
    print(f"XYZ npy shape (valid points): {xyz_coords.shape}")
    pcd = o3d.t.geometry.PointCloud(device)
    # Set the colors for the points
    pcd.point.positions = o3d.core.Tensor(
        xyz_coords, dtype=o3d.core.int32, device=device
    )
    pcd.point.colors = o3d.core.Tensor(
        xyz_colors, dtype=o3d.core.float32, device=device
    )
    o3d.visualization.draw_geometries([pcd.to_legacy()])
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        o3d.t.io.write_point_cloud(args.output_pcd_file, pcd)
        print(f"PCD file save to {args.output_pcd_file}")


if __name__ == "__main__":
    main()
