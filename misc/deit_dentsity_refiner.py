# import torch
#
#
# def get_patches(img, crop_size):
#     _, im_h, im_w = img.shape
#
#     patches1 = []
#     patches1_coords = []
#     patches2 = []
#     patches2_coords = []
#     patches3 = []
#     patches3_coords = []
#     patches8 = []
#     patches8_coords = []
#
#     h_pivot = 8
#     while h_pivot < im_h - 8:
#
#         w_pivot = 8
#         while w_pivot < im_w - 8:
#             y1 = h_pivot if h_pivot + crop_size < im_h - 8 else im_h - crop_size - 8
#             y2 = y1 + crop_size
#             x1 = w_pivot if w_pivot + crop_size < im_w - 8 else im_w - crop_size - 8
#             x2 = x1 + crop_size
#             patches1.append(img[:, y1 - 8:y2 - 8, x1 - 8:x2 - 8])
#             patches1_coords.append((y1 - 8, y2 - 8, x1 - 8, x2 - 8))
#             patches2.append(img[:, y1 - 8:y2 - 8, x1 + 8:x2 + 8])
#             patches2_coords.append((y1 - 8, y2 - 8, x1 + 8, x2 + 8))
#             patches3.append(img[:, y1 + 8:y2 + 8, x1 - 8:x2 - 8])
#             patches3_coords.append((y1 + 8, y2 + 8, x1 - 8, x2 - 8))
#             patches8.append(img[:, y1 + 8:y2 + 8, x1 + 8:x2 + 8])
#             patches8_coords.append((y1 + 8, y2 + 8, x1 + 8, x2 + 8))
#             w_pivot += crop_size
#         h_pivot += crop_size
#
#     return (patches1, patches2, patches3, patches8), \
#            (patches1_coords, patches2_coords, patches3_coords, patches8_coords)
#
#
# def refine_density(model, den, img):
#     crop_size = model.crop_size
#     mask = torch.zeros((crop_size, crop_size))  # Make a mask, we only want the centre 8 x 8 pixels per 32 x 32 block
#     for h in range(crop_size):
#         for w in range(crop_size):
#             if 8 <= h % 32 < 12 and 8 <= w % 32 < 12:
#                 mask[h, w] = 1
#
#     all_patches, all_patches_coords = get_patches(img, crop_size)
#     for patches, patches_coords in zip(all_patches, all_patches_coords):
#         patch_stack = torch.stack(patches).cuda()
#
#         out_den, out_count = model(patch_stack)
#         out_den = out_den.squeeze().cpu()
#
#         for i in range(len(patches_coords)):
#             y1, y2, x1, x2 = patches_coords[i]
#             pred = out_den[i]
#             den[y1:y2, x1:x2] = den[y1:y2, x1:x2] - mask * den[y1:y2, x1:x2]  # Remove old value
#             den[y1:y2, x1:x2] = den[y1:y2, x1:x2] + mask * pred  # Insert new value
#
#     return den



import torch


def get_patches(img, crop_size):
    _, im_h, im_w = img.shape

    patches1 = []
    patches1_coords = []
    patches2 = []
    patches2_coords = []
    patches3 = []
    patches3_coords = []
    patches8 = []
    patches8_coords = []

    h_pivot = 8
    while h_pivot < im_h - 8:

        w_pivot = 8
        while w_pivot < im_w - 8:
            y1 = h_pivot if h_pivot + crop_size < im_h - 8 else im_h - crop_size - 8
            y2 = y1 + crop_size
            x1 = w_pivot if w_pivot + crop_size < im_w - 8 else im_w - crop_size - 8
            x2 = x1 + crop_size
            patches1.append(img[:, y1 - 8:y2 - 8, x1 - 8:x2 - 8])
            patches1_coords.append((y1 - 8, y2 - 8, x1 - 8, x2 - 8))
            patches2.append(img[:, y1 - 8:y2 - 8, x1 + 8:x2 + 8])
            patches2_coords.append((y1 - 8, y2 - 8, x1 + 8, x2 + 8))
            patches3.append(img[:, y1 + 8:y2 + 8, x1 - 8:x2 - 8])
            patches3_coords.append((y1 + 8, y2 + 8, x1 - 8, x2 - 8))
            patches8.append(img[:, y1 + 8:y2 + 8, x1 + 8:x2 + 8])
            patches8_coords.append((y1 + 8, y2 + 8, x1 + 8, x2 + 8))
            w_pivot += crop_size
        h_pivot += crop_size

    return (patches1, patches2, patches3, patches8), \
           (patches1_coords, patches2_coords, patches3_coords, patches8_coords)


def refine_density(model, den, img):
    crop_size = model.crop_size
    mask = torch.zeros((crop_size, crop_size))  # Make a mask, we only want the centre 8 x 8 pixels per 32 x 32 block
    for h in range(crop_size):
        for w in range(crop_size):
            if 8 <= h % 32 < 24 and 8 <= w % 32 < 24:
                mask[h, w] = 1

    all_patches, all_patches_coords = get_patches(img, crop_size)
    for patches, patches_coords in zip(all_patches, all_patches_coords):
        patch_stack = torch.stack(patches).cuda()

        out_den = model(patch_stack)
        out_den = out_den.squeeze().cpu()

        for i in range(len(patches_coords)):
            y1, y2, x1, x2 = patches_coords[i]
            pred = out_den[i]
            den[y1:y2, x1:x2] = den[y1:y2, x1:x2] - mask * den[y1:y2, x1:x2]  # Remove old value
            den[y1:y2, x1:x2] = den[y1:y2, x1:x2] + mask * pred  # Insert new value

    return den