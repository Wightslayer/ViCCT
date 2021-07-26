import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import scipy.io as io
from PIL import Image
import json


def gen_scaled_den_from_points(width, height, gt_points, scale=1., sigma=4, truncate=2):
    """ Creates a density map with given size and the provided points. """

    k = np.zeros((height, width))

    gt_points = scale * gt_points
    for (x, y) in gt_points.astype(int):
        if x < width and y < height:
            k[y, x] = 1  # Note the order of x and y here. Height is stored in first dimension
        else:  # Some datasets have fucked up annotation though. This might happen a lot!
            print("This should never happen!")  # This would mean a head is annotated outside the image.

    density = scipy.ndimage.filters.gaussian_filter(k, sigma, mode='constant', truncate=truncate)  #
    return density


# ============================================================================================= #
#                                           ShanghaiTech                                        #
# ============================================================================================= #
def get_gt_SHT(gt_path, img_w, img_h, scaling=1.):
    mat = io.loadmat(gt_path)
    points = mat["image_info"][0, 0][0, 0][0]
    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                        UCF QNRF ECCV18                                        #
# ============================================================================================= #
def get_gt_UCF_QNRF_ECCV18(gt_path, img_w, img_h, scaling=1.):
    mat = io.loadmat(gt_path)
    points = mat['annPoints']
    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                            LSTN FDST                                          #
# ============================================================================================= #
def LSTN_FDST_regions_to_points(regions):
    n_points = len(regions)

    points = np.ndarray((n_points, 2))
    for idx, region in enumerate(regions):
        shape_attributes = region['shape_attributes']
        x = shape_attributes['x']
        y = shape_attributes['y']
        width = shape_attributes['width']
        height = shape_attributes['height']
        x_centre = x + width / 2
        y_centre = y + height / 2

        points[idx, 0] = x_centre
        points[idx, 1] = y_centre

    return points


def get_gt_LSTN_FDST(gt_path, img_w, img_h, scaling=1.):
    with open(gt_path) as f:
        gt = json.load(f)

    regions = list(gt.values())[0]['regions']
    points = LSTN_FDST_regions_to_points(regions)

    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                           JHU-CROWD++                                         #
# ============================================================================================= #
def JHU_CROWD_lines_to_points(lines):
    n_lines = len(lines)

    points = np.ndarray((n_lines, 2))
    for idx, line in enumerate(lines):
        line_data = line.split(' ')

        points[idx, 0] = line_data[0]
        points[idx, 1] = line_data[1]

    return points


def get_gt_JHU_CROWD_PlusPlus(gt_path, img_w, img_h, scaling=1.):
    with open(gt_path) as f:
        lines = f.readlines()

    points = JHU_CROWD_lines_to_points(lines)

    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                            NWPU-Crowd                                         #
# ============================================================================================= #
def get_gt_NWPU_Crowd(gt_path, img_w, img_h, scaling=1.):
    mat = io.loadmat(gt_path)
    points = mat['annPoints']
    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                            WorldExpo'10                                       #
# ============================================================================================= #
def get_gt_WorldExpo(gt_path, img_w, img_h, scaling=1.):
    """ dens are pre-generated. Image size is provided to comply to standard. Scaling is not supported!"""

    assert scaling == 1., 'Scaling WorldExpo images is not supported!'

    den = pd.read_csv(gt_path, header=None).values

    return den


# ============================================================================================= #
#                                            Municipality                                       #
# ============================================================================================= #
def municipality_csv_to_points(gt_path):
    annotations = pd.read_csv(gt_path)
    xs = annotations['x'].tolist()
    ys = annotations['y'].tolist()

    n_points = len(xs)
    points = np.ndarray((n_points, 2))

    points[:, 0] = xs
    points[:, 1] = ys

    return points


def get_gt_Municipality(gt_path, img_w, img_h, scaling=1.):
    points = municipality_csv_to_points(gt_path)
    den = gen_scaled_den_from_points(img_w, img_h, points, scaling)

    return den


# ============================================================================================= #
#                                          Overall Wrapper                                      #
# ============================================================================================= #

gt_loaders = {
    'SHT': get_gt_SHT,
    'UCF_QNRF_ECCV18': get_gt_UCF_QNRF_ECCV18,
    'LSTN_FDST': get_gt_LSTN_FDST,
    'JHU_CROWD_PlusPlus': get_gt_JHU_CROWD_PlusPlus,
    'NWPU_Crowd': get_gt_NWPU_Crowd,
    'WorldExpo': get_gt_WorldExpo,
    'Municipality': get_gt_Municipality
}


def get_img_and_gt(img_path, gt_path, dataset_type, min_crop_size=224, max_img_size=None):
    img = Image.open(img_path)

    if img.mode == 'L':  # Black and white
        img = img.convert('RGB')  # Colour

    img_w, img_h = img.size

    max_size = max(img_w, img_h)
    # Insert downsizing here
    assert max_img_size is None, 'Capping image size is not yet supported!'

    scale = 1.
    min_size = min(img_w, img_h)
    if min_size < min_crop_size:  # E.g., SHTA has images with less than 224 pixels in some dims.
        scale = min_crop_size / min_size
        new_w = round(scale * img_w)
        new_h = round(scale * img_h)
        img = img.resize((new_w, new_h))
        img_w, img_h = img.size

    den = gt_loaders[dataset_type](gt_path, img_w, img_h, scale)
    den = den.astype(np.float32, copy=False)
    den = Image.fromarray(den)

    return img, den
