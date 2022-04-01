# def zobel_filter(in_raster, size, shape, pre_process):

#%%

import numpy as np
from numba import jit, prange


def zobel_kernel(shape, norm=True, offsets=True, channel_last=True, output_2d=True):

    assert shape[0] == shape[1], "only works for square kernels"
    assert shape[0] in [3, 5, 7], "only works for kernel size 3x3, 5x5 and 7x7"

    if norm == False:
        if shape[0] == 3:
            kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        if shape[0] == 5:
            kernel = np.array(
                [
                    [2, 1, 0, -1, -2],
                    [4, 2, 0, -2, -4],
                    [8, 4, 0, -4, -8],
                    [4, 2, 0, -2, -4],
                    [2, 1, 0, -1, -2],
                ]
            )
        if shape[0] == 7:
            (
                [
                    [4, 2, 1, 0, -1, -2, -4],
                    [8, 4, 2, 0, -2, -4, -8],
                    [16, 8, 4, 0, -4, -8, -16],
                    [32, 16, 8, 0, -8, -16, -32],
                    [16, 8, 4, 0, -4, -8, -16],
                    [8, 4, 2, 0, -2, -4, -8],
                    [4, 2, 1, 0, -1, -2, -4],
                ]
            )

    else:
        if shape[0] == 3:
            kernel = np.array(
                [[0.125, 0.0, -0.125], [0.25, 0.0, -0.25], [0.125, 0.0, -0.125]]
            )
        if shape[0] == 5:
            kernel = np.array(
                [
                    [-0.03333333, -0.06666667, -0.13333333, -0.06666667, -0.03333333],
                    [-0.01666667, -0.03333333, -0.06666667, -0.03333333, -0.01666667],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.01666667, 0.03333333, 0.06666667, 0.03333333, 0.01666667],
                    [0.03333333, 0.06666667, 0.13333333, 0.06666667, 0.03333333],
                ]
            )
        if shape[0] == 7:
            kernel = np.array(
                [
                    [
                        0.01298701,
                        0.00649351,
                        0.00324675,
                        0.0,
                        -0.00324675,
                        -0.00649351,
                        -0.01298701,
                    ],
                    [
                        0.02597403,
                        0.01298701,
                        0.00649351,
                        0.0,
                        -0.00649351,
                        -0.01298701,
                        -0.02597403,
                    ],
                    [
                        0.05194805,
                        0.02597403,
                        0.01298701,
                        0.0,
                        -0.01298701,
                        -0.02597403,
                        -0.05194805,
                    ],
                    [
                        0.1038961,
                        0.05194805,
                        0.02597403,
                        0.0,
                        -0.02597403,
                        -0.05194805,
                        -0.1038961,
                    ],
                    [
                        0.05194805,
                        0.02597403,
                        0.01298701,
                        0.0,
                        -0.01298701,
                        -0.02597403,
                        -0.05194805,
                    ],
                    [
                        0.02597403,
                        0.01298701,
                        0.00649351,
                        0.0,
                        -0.00649351,
                        -0.01298701,
                        -0.02597403,
                    ],
                    [
                        0.01298701,
                        0.00649351,
                        0.00324675,
                        0.0,
                        -0.00324675,
                        -0.00649351,
                        -0.01298701,
                    ],
                ]
            )

    idx_offsets = []
    # weights = []
    if offsets:

        if len(kernel.shape) == 2:

            offset_shape = [1, kernel.shape[0], kernel.shape[1]]
            offset_kernel = np.zeros(offset_shape, dtype="float32")

        for z in range(offset_kernel.shape[0]):
            for x in range(offset_kernel.shape[1]):
                for y in range(offset_kernel.shape[2]):
                    # current_weight = kernel[z][x][y]

                    # if remove_zero_weights and current_weight == 0.0:
                    #     continue

                    if channel_last:
                        if output_2d:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
                        else:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                    z - (offset_kernel.shape[0] // 2),
                                ]
                            )
                    else:
                        if output_2d:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
                        else:
                            idx_offsets.append(
                                [
                                    z - (offset_kernel.shape[0] // 2),
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
                    # weights.append(current_weight)

    # if channel_last:
    #     kernel = kernel.reshape(kernel.shape[1], kernel.shape[2], kernel.shape[0])

    # if output_2d and channel_last:
    #     kernel = kernel[:, :, 0]
    # elif output_2d:
    #     kernel = kernel[0, :, :]

    if offsets:
        return (
            kernel,
            np.array(idx_offsets, dtype=int),
            # np.array(weights, dtype=float),
        )

    return kernel


# # TODO deal with flatten


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def convolve_sobel_2D(arr, kernel, offsets, pre_process=False, pre_process_kernel=None):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1

    hood_size = len(offsets)

    result = np.empty_like(arr)

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_adj:
                    offset_x = x_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_adj:
                    offset_y = y_adj

                hood_values[n] = arr[offset_x, offset_y]

            # if pre_process == True:

            #     hood_values = hood_values * pre_process_kernel

            transformation = hood_values * kernel

            result[x, y] = np.sum(transformation)

    return result


def zobel_filter(arr, size=[3, 3], normalised_sobel=False, gaussian_preprocess=False):
    sobel_filter, offsets = zobel_kernel(size, norm=normalised_sobel)
    sobel_filter90 = np.rot90(sobel_filter, 3)
    sobel_flattened = sobel_filter.flatten()
    sobel_flattened90 = sobel_filter90.flatten()
    res1 = convolve_sobel_2D(
        arr, sobel_flattened, offsets, pre_process=False, pre_process_kernel=None
    )
    res2 = convolve_sobel_2D(
        arr, sobel_flattened90, offsets, pre_process=False, pre_process_kernel=None
    )

    filtered = (res1 ** 2 + res2 ** 2) ** 0.5
    return filtered

#%%
import sys


yellow_path = "//wsl$/Ubuntu-20.04/home/afer/yellow/"
# buteo_buteo_follow = "D:/buteo/buteo/"

import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/')


# sys.path.append(buteo_follow)
# sys.path.append(buteo_buteo_follow)
# sys.path.append(buteo_buteo_follow + "filters/")
# sys.path.append(buteo_buteo_follow + "machine_learning/")
# sys.path.append(buteo_buteo_follow + "raster/")

import time

start = time.time()


from convolutions import *
from kernel_generator import *
from filter import *
# from patch_extraction import *
from raster import *
from raster.io import *
from osgeo import gdal

ref = r"V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dtm\DTM_1km_6052_661.tif"
out = r"V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\hat\SOBELDTM_1km_6052_661_.tif"
raster = gdal.Open(ref)
bandarr = raster.GetRasterBand(1).ReadAsArray()
npy = np.array(bandarr)

result = zobel_filter(
    npy, size=[5, 5], normalised_sobel=False, gaussian_preprocess=False
)

array_to_raster(result, reference=ref, out_path=out)

end = time.time()
print(end - start)


# %%
