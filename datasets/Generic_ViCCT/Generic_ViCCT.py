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


def get_n_items_in_split(dataset):
    """ Reads the dataset csv file from disk and return the number of entries in this files. """

    data_split_path = dataset['split_to_use_path']
    data_split = pd.read_csv(data_split_path)
    return data_split.size


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

        # A split can have to elements. Add a dummy item so that dataloader can be created.
        if len(datasets) == 0:
            self.data_files = [
                ('Dummy', 'Dummy', 'Dummy')]  # Handy for testing on a separate test set that doesn't have a train set
            print(f'No items added for {mode} split.')
            self.num_samples = 0
            return

        print(f'Constructing the {mode} split:')

        static_datasets = []
        dynamic_datasets = []  # These are the datasets that should occupy X% of the samples
        percentage_dynamic = 0
        for dataset in datasets:
            assert 'percent_of_split' not in dataset or 'n_copies' not in dataset, \
                'percent_of_split and n_copies cannot be used together!'

            if 'percent_of_split' in dataset:
                dynamic_datasets.append(dataset)
                percentage_dynamic += dataset['percent_of_split']
            else:
                static_datasets.append(dataset)
        percentage_static = 100 - percentage_dynamic

        assert len(static_datasets) > 0, 'Must provide at least one non-dynamic dataset'

        # Add the non-dynamic datasets to the data files
        for dataset in static_datasets:
            if 'n_copies' in dataset:
                n_copies = dataset['n_copies']
            else:
                n_copies = 1
            self.add_dataset_to_data_files(dataset, n_copies)

        images_per_percent = len(self.data_files) / percentage_static

        for dataset in dynamic_datasets:

            # Compute how many times we need to add the dataset to (approx) reach the desired percentage of whole split.
            percent_of_split = dataset['percent_of_split']
            entries_needed = images_per_percent * percent_of_split
            n_links = get_n_items_in_split(dataset)
            n_copies = round(entries_needed / n_links)
            n_copies = max(n_copies, 1)  # n_copies could be 0, but we want at least one copy in our data files

            # Compute and print the actual percentage of whole now that we know how many copies to add.
            actual_percentage = n_copies * n_links / images_per_percent
            dataset_name = dataset['dataset_name']
            print(f'  <<<dataset {dataset_name} will be {actual_percentage:.3f}% of the whole split.>>>')

            # Add the dataset n times to the datafiles to reach desired split
            self.add_dataset_to_data_files(dataset, n_copies)

        self.num_samples = len(self.data_files)
        print(f'{len(self.data_files)} {self.mode} images found in {len(datasets)} datasets.')

    def add_dataset_to_data_files(self, dataset, n_copies):
        """ Adds a dataset to the data files. """

        dataset_name = dataset['dataset_name']
        base_path = dataset['dataset_path']
        den_gen_key = dataset['den_gen_key']
        data_split_path = dataset['split_to_use_path']
        data_split = pd.read_csv(data_split_path)
        data_split = data_split.to_numpy()

        if n_copies > 1:
            for copy_number in range(n_copies):
                extended_name = dataset_name + '_copy' + str(copy_number + 1)
                self.make_all_links(extended_name, base_path, data_split, den_gen_key)
        else:
            self.make_all_links(dataset_name, base_path, data_split, den_gen_key)

    def make_all_links(self, dataset_name, base_path, data_split, den_gen_key):
        """ Combines the relative paths to the base path and adds them to the data files, along with the key which
            specifies the appropriate function to generate the GT density maps. """

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
