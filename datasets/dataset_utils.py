import math
import torch
import scipy.ndimage
import numpy as np


def img_equal_split(img, crop_size, overlap):
    """
    Splits the image into crops, where all crops have equal overlap to adjacent crops.
    :param img: The image to split into crops. Can also be the density map
    :param crop_size: Crops are of shape 'n_channels x crops_size x crop_size'. n_channels is inferred from image.
    :param overlap: AT LEAST this many pixels of overlap between adjacent crops.
    :return: A stack containing all crops. Shape: [n_crops, n_channels, crops_size, crop_size]
    """

    channels, h, w = img.shape

    n_cols = (w - crop_size) / (crop_size - overlap) + 1
    n_cols = math.ceil(n_cols)  # At least this many crops needed to get >= overlap pixels of overlap
    n_rows = (h - crop_size) / (crop_size - overlap) + 1
    n_rows = math.ceil(n_rows)  # At least this many crops needed to get >= overlap pixels of overlap

    if n_cols > 1:
        overlap_w = crop_size - (w - crop_size) / (n_cols - 1)
        overlap_w = math.floor(overlap_w)
    else:  # edge case (SHTA)
        overlap_w = 0

    if n_rows > 1:
        overlap_h = crop_size - (h - crop_size) / (n_rows - 1)
        overlap_h = math.floor(overlap_h)
    else:  # edge case (SHTA)
        overlap_h = 0

    crops = torch.zeros((n_rows * n_cols, channels, crop_size, crop_size))

    for r in range(n_rows):
        for c in range(n_cols):
            y1 = r * (crop_size - overlap_h) if r * (crop_size - overlap_h) + crop_size <= h else h - crop_size
            y2 = y1 + crop_size
            x1 = c * (crop_size - overlap_w) if c * (crop_size - overlap_w) + crop_size <= w else w - crop_size
            x2 = x1 + crop_size

            item_idx = r * n_cols + c
            crops[item_idx, :, :, :] = img[:, y1:y2, x1:x2]

    return crops


def img_equal_unsplit(crops, overlap, ignore_buffer, img_h, img_w, img_channels):
    """
    Unsplits the image split with 'img_equal_split'
    :param crops: A stack of crops that make up an entire image/density map
    :param overlap: The overlap used to split the image in 'img_equal_split'
    :param ignore_buffer: How many pixels to ignore at the corner of crops. Image borders are never ignored.
    :param img_h: Height of the image
    :param img_w: Width of the image
    :param img_channels: Number of channels of the image. E.g. 3 for RGB, 1 for density maps.
    :return: The reconstructed image. Overlap is resolved by taking the average of overlapping pixels
    """

    w, h = img_w, img_h
    crop_size = crops.shape[-1]
    n_cols = (w - crop_size) / (crop_size - overlap) + 1
    n_cols = math.ceil(n_cols)  # At least this many crops needed to get >= overlap pixels of overlap
    n_rows = (h - crop_size) / (crop_size - overlap) + 1
    n_rows = math.ceil(n_rows)  # At least this many crops needed to get >= overlap pixels of overlap

    if n_cols > 1:
        overlap_w = crop_size - (w - crop_size) / (n_cols - 1)
        overlap_w = math.floor(overlap_w)
    else:
        overlap_w = 0

    if n_rows > 1:
        overlap_h = crop_size - (h - crop_size) / (n_rows - 1)
        overlap_h = math.floor(overlap_h)
    else:
        overlap_h = 0

    new_img = torch.zeros((img_channels, h, w))
    divider = torch.zeros((img_channels, h, w))

    for r in range(n_rows):
        for c in range(n_cols):
            y1 = r * (crop_size - overlap_h) if r * (crop_size - overlap_h) + crop_size <= h else h - crop_size
            y2 = y1 + crop_size
            x1 = c * (crop_size - overlap_w) if c * (crop_size - overlap_w) + crop_size <= w else w - crop_size
            x2 = x1 + crop_size

            ign_top = ignore_buffer if r != 0 else 0
            ign_bot = ignore_buffer if r != n_rows - 1 else 0
            ign_left = ignore_buffer if c != 0 else 0
            ign_right = ignore_buffer if c != n_cols - 1 else 0

            item_idx = r * n_cols + c
            new_img[:, y1 + ign_top:y2 - ign_bot, x1 + ign_left:x2 - ign_right] += \
                crops[item_idx, :, 0 + ign_top:crop_size - ign_bot, 0 + ign_left:crop_size - ign_right]
            divider[:, y1 + ign_top:y2 - ign_bot, x1 + ign_left:x2 - ign_right] += 1

    return new_img / divider
