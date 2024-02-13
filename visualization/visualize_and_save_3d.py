"""
Visualize 3D shape with 1 or 0 values. XYZ left hand.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
        "--output_obj_file", required=False, type=str, help="Output obj filepath"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    print("Loading volume")
    volume = np.load(args.input_file)
    print(f"3D Shape num_slices x 2D height x 2D width (zyx): {volume.shape}")

    rescaled_volume = zoom(volume, (0.1, 0.1, 0.1))
    print(f"Rescaled 3D shape num_slices x 2D height x 2D width (zyx): {rescaled_volume.shape}")
    del volume

    # FIXME: vertices are not enough
    x_list = []
    y_list = []
    z_list = []
    if args.output_obj_file:
        print(f"Saving obj file to {args.output_obj_file}")
        with open(args.output_obj_file, 'w') as out_obj:
            for z in range(rescaled_volume.shape[0]):
                for y in range(rescaled_volume.shape[1]):
                    for x in range(rescaled_volume.shape[2]):
                        if rescaled_volume[z][y][x] > 0.0:
                            out_obj.write(f"v {x} {y} {z}\n")
                            x_list.append(x)
                            y_list.append(y)
                            z_list.append(z)
        print(f"obj file saved to {args.output_obj_file}")

    # Color by red the valid pixels (1.0 values)
    # print("Visualizing 3D volume as voxels")
    # colors = np.empty(rescaled_volume.shape, dtype=object)
    # colors[rescaled_volume > 0.0] = 'red'
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.voxels(rescaled_volume, facecolors=colors)
    # plt.show()

    print("Visualizing 3D volume as point cloud")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # TODO: Draw smaller points
    ax.scatter(x_list, y_list, z_list, linewidths=0.1)
    plt.show()


if __name__ == "__main__":
    main()
