import argparse

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two 3D shapes",
    )
    parser.add_argument(
        "--label_file",
        required=True,
        type=str,
        help="Ground truth numpy array 3D shaped filepath",
    )
    parser.add_argument(
        "--pred_file",
        required=True,
        type=str,
        help="Ground truth numpy array 3D shaped filepath",
    )
    parser.add_argument(
        "--rescale_factor",
        required=True,
        type=float,
        help="Rescale factor to avoid OOM, e.g. 0.1",
    )
    parser.add_argument(
        "--output_pcd_file", required=False, type=str, help="Output pcd filepath"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    print(f"Loading label volume from {args.label_file}")
    label_volume_xyz = np.load(args.label_file)
    print(
        f"Loaded label 3D shape 2D width x 2D height X num_slices (xyz): {label_volume_xyz.shape}"
    )
    rescaled_label_volume = zoom(
        label_volume_xyz,
        (args.rescale_factor, args.rescale_factor, args.rescale_factor),
    )
    print(
        f"Rescaled label 3D shape 2D width x 2D height X num_slices (xyz): {rescaled_label_volume.shape}"
    )
    del label_volume_xyz

    print(f"Loading prediction volume from {args.pred_file}")
    pred_volume_xyz = np.load(args.pred_file)
    print(
        f"Loaded pred 3D shape 2D width x 2D height X num_slices (xyz): {pred_volume_xyz.shape}"
    )
    rescaled_pred_volume = zoom(
        pred_volume_xyz, (args.rescale_factor, args.rescale_factor, args.rescale_factor)
    )
    print(
        f"Rescaled pred 3D shape 2D width x 2D height X num_slices (xyz): {rescaled_pred_volume.shape}"
    )
    del pred_volume_xyz

    assert rescaled_pred_volume.shape == rescaled_label_volume.shape

    # Draw point cloud with green TP, blue FN, red FP
    tp_xyz_coords = np.argwhere(
        (rescaled_pred_volume > 0.0) & (rescaled_label_volume > 0.0)
    )
    fn_xyz_coords = np.argwhere(
        (rescaled_pred_volume == 0.0) & (rescaled_label_volume > 0.0)
    )
    fp_xyz_coords = np.argwhere(
        (rescaled_pred_volume > 0.0) & (rescaled_label_volume == 0.0)
    )
    tp_xyz_colors = np.zeros_like(tp_xyz_coords)
    tp_xyz_colors[:, :] = [0.0, 1.0, 0.0]
    fn_xyz_colors = np.zeros_like(fn_xyz_coords)
    fn_xyz_colors[:, :] = [0.0, 0.0, 1.0]
    fp_xyz_colors = np.zeros_like(fp_xyz_coords)
    fp_xyz_colors[:, :] = [1.0, 0.0, 0.0]

    xyz_coords = np.concatenate([tp_xyz_coords, fn_xyz_coords, fp_xyz_coords], axis=0)
    xyz_colors = np.concatenate([tp_xyz_colors, fn_xyz_colors, fp_xyz_colors], axis=0)
    # Save point cloud to PCD format, o3d visualizer is faster than matplotlib
    print(f"XYZ npy shape: {xyz_coords.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_coords)
    pcd.colors = o3d.utility.Vector3dVector(xyz_colors)
    o3d.visualization.draw_geometries([pcd])
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        o3d.io.write_point_cloud(args.output_pcd_file, pcd)
        print(f"PCD file save to {args.output_pcd_file}")


if __name__ == "__main__":
    main()
