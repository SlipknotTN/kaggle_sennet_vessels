import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two 3D shapes",
    )
    parser.add_argument(
        "--label_file", required=True, type=str, help="Ground truth numpy array 3D shaped filepath"
    )
    parser.add_argument(
        "--pred_file", required=True, type=str, help="Ground truth numpy array 3D shaped filepath"
    )
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    print("Loading label volume")
    label_volume = np.load(args.label_file)
    print(f"Loaded label 3D shape num_slices x 2D height x 2D width (zyx): {label_volume.shape}")
    rescaled_label_volume = zoom(label_volume, (0.1, 0.1, 0.1))
    print(f"Rescaled label 3D shape num_slices x 2D height x 2D width (zyx): {rescaled_label_volume.shape}")
    del label_volume

    print("Loading prediction volume")
    pred_volume = np.load(args.pred_file)
    print(f"Loaded pred 3D shape num_slices x 2D height x 2D width (zyx): {pred_volume.shape}")
    rescaled_pred_volume = zoom(pred_volume, (0.1, 0.1, 0.1))
    print(f"Rescaled pred 3D shape num_slices x 2D height x 2D width (zyx): {rescaled_pred_volume.shape}")
    del pred_volume

    assert rescaled_pred_volume.shape == rescaled_label_volume.shape

    x_tp_list = []
    y_tp_list = []
    z_tp_list = []
    x_fp_list = []
    y_fp_list = []
    z_fp_list = []
    x_fn_list = []
    y_fn_list = []
    z_fn_list = []
    for z in range(rescaled_label_volume.shape[0]):
        for y in range(rescaled_label_volume.shape[1]):
            for x in range(rescaled_label_volume.shape[2]):
                if rescaled_pred_volume[z][y][x] > 0.0 and rescaled_label_volume[z][y][x] > 0.0:
                    x_tp_list.append(x)
                    y_tp_list.append(y)
                    z_tp_list.append(z)
                if rescaled_pred_volume[z][y][x] > 0.0 and rescaled_label_volume[z][y][x] == 0.0:
                    x_fp_list.append(x)
                    y_fp_list.append(y)
                    z_fp_list.append(z)
                if rescaled_pred_volume[z][y][x] == 0.0 and rescaled_label_volume[z][y][x] > 0.0:
                    x_fn_list.append(x)
                    y_fn_list.append(y)
                    z_fn_list.append(z)

    print("Visualizing 3D volume comparison as voxels")
    colors = np.empty(rescaled_pred_volume.shape, dtype=object)
    # TP
    colors[rescaled_pred_volume > 0.0 and rescaled_label_volume > 0.0] = 'green'
    # FP
    colors[rescaled_pred_volume > 0.0 and rescaled_label_volume == 0.0] = 'red'
    # FN
    colors[rescaled_pred_volume == 0.0 and rescaled_label_volume > 0.0] = 'blue'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(np.max(rescaled_pred_volume, rescaled_label_volume), facecolors=colors)
    plt.show()

    print("Visualizing 3D volume as point cloud")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # TODO: Draw smaller points
    ax.scatter(x_tp_list, y_tp_list, z_tp_list, c="green")
    ax.scatter(x_fp_list, y_fp_list, z_fp_list, c="red")
    ax.scatter(x_fn_list, y_fn_list, z_fn_list, c="blue")
    plt.show()


if __name__ == "__main__":
    main()
