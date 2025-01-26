"""
Visualize 3D point clouds and save them to PCD format:
- Load full resolution 3D numpy point clouds
- Rescale it to reduce memory footprint
- Visualize it
- [Optional] Save the rescaled version in pcd format

It can be used for both labels and predictions.

Coordinates system: xyz right-handed, z-up.
Slices (z) going from bottom (0) to up (max = num_slices - 1)

Example with rescaling:
    python visualization/visualize_3d_npy_and_save_pcd.py \
    --input_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_prediction_xyz.npy \
    --output_pcd_file ./outputs/submissions/v48/kidney_3_sparse_tta_5max_thresh_0.1_1024/3d/kidney_3_sparse_prediction_rescaled_0.1_xyz.pcd \
    --rescale_factor 0.1

Example without rescaling:
    python visualization/visualize_3d_npy_and_save_pcd.py \
    --input_file /media/michele/DATA-2/Datasets/kaggle_vessels/3D/train/kidney_3_sparse/kidney_3_sparse_label_xyz.npy \
    --output_pcd_file /media/michele/DATA-2/Datasets/kaggle_vessels/3D/train/kidney_3_sparse/kidney_3_sparse_label_xyz.pcd
"""
import argparse

import numpy as np
import open3d as o3d
from scipy.ndimage import zoom


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize and save 3D point clouds",
    )
    parser.add_argument(
        "--input_file", required=True, type=str, help="Numpy array 3D shaped filepath"
    )
    parser.add_argument(
        "--output_pcd_file", required=False, type=str, help="Output pcd filepath"
    )
    parser.add_argument(
        "--rescale_factor",
        required=False,
        default=1.0,
        type=float,
        help="Rescale factor to avoid OOM, e.g. 0.1",
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

    print(f"Loading volume from {args.input_file}")
    volume_xyz = np.load(args.input_file)
    print(f"Loaded 3D shape width x height X num_slices (xyz): {volume_xyz.shape}")

    if args.rescale_factor < 1.0:
        print("Rescaling numpy volume")
        rescaled_volume = zoom(
            volume_xyz, (args.rescale_factor, args.rescale_factor, args.rescale_factor)
        )
        print(
            f"Rescaled 3D shape width x height X num_slices (xyz): {rescaled_volume.shape}"
        )
        del volume_xyz
    else:
        print("No rescaling")
        rescaled_volume = volume_xyz

    # Create the point cloud according to the rescaled numpy 3D volume
    pcd = o3d.t.geometry.PointCloud(device)
    # Get the coordinates for the valid points (shape is num_points x 3)
    points_coords = np.argwhere(rescaled_volume > 0.0)
    print(f"Point coordinates npy shape (valid points): {points_coords.shape}")
    points_colors = np.zeros_like(points_coords, dtype=np.float32)
    # Set the grayscale color for each coordinate according to the confidence
    # 0 -> white (1.0, 1.0, 1.0), 1 -> black (0.0, 0.0, 0.0)
    # Save intensities metadata as well to keep track of the confidence value
    intensities = np.zeros(shape=(points_colors.shape[0], 1))
    for idx, point_coord in enumerate(points_coords):
        confidence = float(rescaled_volume[tuple(point_coord)])
        intensities[idx] = confidence
        points_colors[idx, :] = (1.0 - confidence) * np.array([1.0, 1.0, 1.0])
    pcd.point.positions = o3d.core.Tensor(
        points_coords, dtype=o3d.core.int32, device=device
    )
    pcd.point.colors = o3d.core.Tensor(
        points_colors, dtype=o3d.core.float32, device=device
    )
    pcd.point.intensities = o3d.core.Tensor(
        intensities, dtype=o3d.core.float32, device=device
    )
    # Visualize the pcd: converting to legacy (no Tensor) is necessary
    o3d.visualization.draw_geometries([pcd.to_legacy()])

    # Save point cloud to PCD format, o3d visualizer is faster than matplotlib
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        o3d.t.io.write_point_cloud(args.output_pcd_file, pcd)
        print(f"PCD file save to {args.output_pcd_file}")


if __name__ == "__main__":
    main()
