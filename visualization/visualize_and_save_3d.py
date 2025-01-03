"""
Visualize 3D shape with 1 or 0 values.

Coordinate system: xyz right handed, z-up.
Slices (z) going from bottom (0) to up (max = num_slices - 1)
"""
import argparse

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize and save 3D shape",
    )
    parser.add_argument(
        "--input_file", required=True, type=str, help="Numpy array 3D shaped filepath"
    )
    parser.add_argument(
        "--output_pcd_file", required=False, type=str, help="Output pcd filepath"
    )
    parser.add_argument(
        "--rescale_factor",
        required=True,
        type=float,
        help="Rescale factor to avoid OOM, e.g. 0.1",
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    print(f"Loading volume from {args.input_file}")
    volume_xyz = np.load(args.input_file)
    print(
        f"Loaded 3D shape 2D width x 2D height X num_slices (xyz): {volume_xyz.shape}"
    )

    print("Rescaling volume")
    rescaled_volume = zoom(
        volume_xyz, (args.rescale_factor, args.rescale_factor, args.rescale_factor)
    )
    print(
        f"Rescaled 3D shape 2D width x 2D height X num_slices (xyz): {rescaled_volume.shape}"
    )
    del volume_xyz

    # Ground truth 3D points (1.0 value), set RED RGB color
    xyz_coords = np.argwhere(rescaled_volume > 0.0)
    xyz_colors = np.zeros_like(xyz_coords)
    xyz_colors[:, :] = [1.0, 0.0, 0.0]

    # Save point cloud to PCD format, o3d visualizer is faster than matplotlib
    print(f"XYZ npy shape: {xyz_coords.shape}")
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_coords)
        pcd.colors = o3d.utility.Vector3dVector(xyz_colors)
        o3d.io.write_point_cloud(args.output_pcd_file, pcd)
        print(f"PCD file save to {args.output_pcd_file}")
        o3d.visualization.draw_geometries([pcd])
        # To save custom intensity
        # pcd.point["positions"] = o3d.core.Tensor(xyz)
        # pcd.point["intensities"] = o3d.core.Tensor(i)


if __name__ == "__main__":
    main()
