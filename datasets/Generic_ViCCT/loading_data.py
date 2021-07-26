import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import datasets.transforms as own_transforms

from .settings import cfg_data
from .Generic_ViCCT import Generic_ViCCT


def loading_data(crop_size):
    # train transforms

    train_main_transform = own_transforms.Compose([
        own_transforms.RandomCrop([crop_size, crop_size]),  # For training, crop a random part of an image (and GT)
        own_transforms.RandomHorizontallyFlip()  # Randomly flips both image and GT
    ])

    train_img_transform = standard_transforms.Compose([  # Transforms to apply. Can also add augmentations here
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    val_img_transform = standard_transforms.Compose([  # Don't apply augmentations to eval data. Thus new transform.
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    gt_transform = standard_transforms.Compose([  # To scale the GT density map by a given amount
        own_transforms.LabelScale(cfg_data.LABEL_FACTOR)
    ])

    train_cropper = own_transforms.Compose([  # Redundant perhaps? Unless global augmentation is applied
        own_transforms.RandomTensorCrop([crop_size, crop_size])
    ])

    restore_transform = standard_transforms.Compose([  # To get back the original image
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])

    train_set = Generic_ViCCT(cfg_data.TRAIN_DATASETS, 'train', crop_size,
                              main_transform=train_main_transform,
                              img_transform=train_img_transform,
                              gt_transform=gt_transform,
                              cropper=train_cropper)
    train_loader = DataLoader(train_set,
                              batch_size=cfg_data.TRAIN_BS,
                              num_workers=cfg_data.N_WORKERS,
                              shuffle=True, drop_last=True)

    val_set = Generic_ViCCT(cfg_data.VAL_DATASETS, 'val', crop_size,
                            main_transform=None,
                            img_transform=val_img_transform,
                            gt_transform=gt_transform,
                            cropper=None)
    val_loader = DataLoader(val_set,
                            batch_size=cfg_data.VAL_BS,
                            num_workers=cfg_data.N_WORKERS,
                            shuffle=False, drop_last=False)

    test_set = Generic_ViCCT(cfg_data.TEST_DATASETS, 'test', crop_size,
                             main_transform=None,
                             img_transform=val_img_transform,
                             gt_transform=gt_transform,
                             cropper=None)
    test_loader = DataLoader(test_set,
                             batch_size=cfg_data.VAL_BS,
                             num_workers=cfg_data.N_WORKERS,
                             shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, restore_transform
