"""
Visualize 3D shape with 1 or 0 values. XYZ left hand.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    # TODO: Add rescale factor
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    print("Loading volume")
    volume = np.load(args.input_file)
    print(f"Loaded 3D Shape num_slices x 2D height x 2D width (zyx): {volume.shape}")

    print("Rescaling volume")
    rescaled_volume = zoom(volume, (0.25, 0.25, 0.25))
    print(f"Rescaled 3D shape num_slices x 2D height x 2D width (zyx): {rescaled_volume.shape}")
    del volume

    # TODO: Use np.indices?!
    x_list = []
    y_list = []
    z_list = []
    xyz_values = []
    for z in range(rescaled_volume.shape[0]):
        for y in range(rescaled_volume.shape[1]):
            for x in range(rescaled_volume.shape[2]):
                if rescaled_volume[z][y][x] > 0.0:
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)
                    xyz_values.append([x, y, z])

    # Save point cloud to PCD format
    xyz_npy = np.array(xyz_values)
    print(f"XYZ npy shape: {xyz_npy.shape}")
    if args.output_pcd_file:
        print(f"Saving PCD file to {args.output_pcd_file}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_npy)
        o3d.io.write_point_cloud(args.output_pcd_file, pcd)
        print(f"PCD file save to {args.output_pcd_file}")
        o3d.visualization.draw_geometries([pcd])
        # To save custom intensity instead of increasing from bottom to top
        # pcd.point["positions"] = o3d.core.Tensor(xyz)
        # pcd.point["intensities"] = o3d.core.Tensor(i)

    # Lighter and quicker visualization than voxels
    print("Visualizing 3D volume as point cloud")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_list, y_list, z_list, s=5)
    plt.show()


if __name__ == "__main__":
    main()
