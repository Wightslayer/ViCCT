import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .Generic_ViCCT import Generic_ViCCT


def loading_data(crop_size):

    # <<<<<<<<<<<<<<<<<<<<  Stuff for the training dataloader  >>>>>>>>>>>>>>>>>>>> #
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop([crop_size, crop_size]),  # For training, crop a random part of an image (and GT)
        own_transforms.RandomHorizontallyFlip()  # Randomly flips both image and GT
    ])

    train_transforms = []  # A list to which we append all transformations for images (for training only)
    if cfg_data.USE_GAMMA_TRANSFORM:  # If USE_GAMMA_TRANSFORM in settings.py is True
        train_transforms.append(own_transforms.RandomGammaTransform())

    if cfg_data.USE_GRAYSCALE_TRANSFORM:  # If USE_GRAYSCALE_TRANSFORM in settings.py is True
        train_transforms.append(own_transforms.RandomGrayscale())

    train_transforms.append(standard_transforms.ToTensor())  # Transforms PIL image to PyTorch Tensor
    train_transforms.append(standard_transforms.Normalize(*cfg_data.MEAN_STD))  # Normalises the image

    train_img_transform = standard_transforms.Compose(train_transforms)

    # <<<<<<<<<<<<<<<<<<<<  Stuff for the training and testing dataloader  >>>>>>>>>>>>>>>>>>>> #
    val_img_transform = standard_transforms.Compose([  # Don't apply augmentations to eval data. Thus new transform.
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    # <<<<<<<<<<<<<<<<<<<<  Stuff for all dataloaders  >>>>>>>>>>>>>>>>>>>> #
    gt_transform = standard_transforms.Compose([  # To scale the GT density map by a given amount
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    # <<<<<<<<<<<<<<<<<<<<  Restore transform so we can visualise actual image  >>>>>>>>>>>>>>>>>>>> #
    restore_transform = standard_transforms.Compose([  # To get back the original image
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])

# ===================================================================== #
#                              TRAIN DATALOADER                         #
# ===================================================================== #

    train_set = Generic_ViCCT(cfg_data.TRAIN_DATASETS, 'train', crop_size,
                              main_transform=train_main_transform,
                              img_transform=train_img_transform,
                              gt_transform=gt_transform)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

# ===================================================================== #
#                               VAL DATALOADER                          #
# ===================================================================== #
    val_set = Generic_ViCCT(cfg_data.VAL_DATASETS, 'val', crop_size,
                            main_transform=None,
                            img_transform=val_img_transform,
                            gt_transform=gt_transform)
    val_loader = DataLoader(val_set,
                            batch_size=cfg_data.VAL_BS,
                            num_workers=cfg_data.N_WORKERS,
                            shuffle=False, drop_last=False)

# ===================================================================== #
#                              TEST DATALOADER                          #
# ===================================================================== #
    test_set = Generic_ViCCT(cfg_data.TEST_DATASETS, 'test', crop_size,
                             main_transform=None,
                             img_transform=val_img_transform,
                             gt_transform=gt_transform)
    test_loader = DataLoader(test_set,
                             batch_size=cfg_data.VAL_BS,
                             num_workers=cfg_data.N_WORKERS,
                             shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, restore_transform
