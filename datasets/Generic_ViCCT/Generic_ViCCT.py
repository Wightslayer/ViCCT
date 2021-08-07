import os

import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils import data

from PIL import Image
from .settings import cfg_data
from datasets.data_retriever_and_generator import get_img_and_gt
from datasets.dataset_utils import img_equal_split, img_equal_unsplit


class Generic_ViCCT(data.Dataset):
    def __init__(self, datasets, mode, crop_size,
                 main_transform=None, img_transform=None, gt_transform=None, cropper=None):

        self.crop_size = crop_size  # 224
        self.mode = mode  # train, test or eval

        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.cropper = cropper

        self.data_files = []

        print()  # Newline
        print(f'Constructing combined {self.mode} dataset')
        for dataset in datasets:
            dataset_name = dataset['dataset_name']
            base_path = dataset['dataset_path']
            den_gen_key = dataset['den_gen_key']
            data_split_path = dataset['split_to_use_path']
            data_split = pd.read_csv(data_split_path)
            data_split = data_split.to_numpy()

            if 'n_copies' in dataset:
                n_copies = dataset['n_copies']
                for copy_number in range(n_copies):
                    extended_name = dataset_name + '_copy' + str(copy_number + 1)
                    self.add_dataset_to_data_files(extended_name, base_path, data_split, den_gen_key)
            else:
                self.add_dataset_to_data_files(dataset_name, base_path, data_split, den_gen_key)

        if not self.data_files:  # If we only have a train or test set, we can still initialize the dataloader.
            self.data_files = [
                ('Dummy', 'Dummy', 'Dummy')]  # Handy for testing on a separate test set that doesn't have a train set.
        self.num_samples = len(self.data_files)

        if self.data_files[0] == 'Dummy':
            print(f'No {self.mode} images found in {len(datasets)} datasets.')
        else:
            print(f'{len(self.data_files)} {self.mode} images found in {len(datasets)} datasets.')

    def add_dataset_to_data_files(self, dataset_name, base_path, data_split, den_gen_key):

        for rel_img_path, rel_gt_path in data_split:
            abs_img_path = os.path.join(base_path, rel_img_path)
            abs_gt_path = os.path.join(base_path, rel_gt_path)
            self.data_files.append((abs_img_path, abs_gt_path, den_gen_key))

        n_imgs = len(data_split)
        print(f'  Added dataset "{dataset_name}" with {n_imgs} images')

    def __getitem__(self, index):
        """ Get img and gt stored at index 'index' in data files. """
        img, den = self.read_image_and_gt(index)

        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        if self.mode == 'train':
            img_crop, den_crop = self.cropper(img, den.unsqueeze(0))
            return img_crop, den_crop
        else:
            img_stack = img_equal_split(img, self.crop_size, cfg_data.OVERLAP)
            gts_stack = img_equal_split(den.unsqueeze(0), self.crop_size, cfg_data.OVERLAP)
            return img, img_stack, gts_stack

    def read_image_and_gt(self, index):
        """
        Retrieves the image and density map from the disk.
        :param index: Index of data_files.
        :return: image and gt density map as PIL Images
        """

        img_path, gt_path, get_gt_key = self.data_files[index]
        img, den = get_img_and_gt(img_path, gt_path, get_gt_key)

        return img, den

    def __len__(self):
        """ The number of paths stored in data files. """
        return self.num_samples
