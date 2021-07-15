# See LICENCE for copyrights

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

import models.ViCCTModels  # Needed to register models for 'create_model'
from timm.models import create_model

import importlib

from trainer import Trainer
from config import cfg
from shutil import copyfile
import random


# We initialise the DeiT encoder part with pretrained weights. The ViCCT extension is randomly initialised.
model_mapping = {
    # 'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    'ViCCT_tiny': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    # 'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    'ViCCT_small': 'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    # 'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
    'ViCCT_base': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    'ViCCT_base_384': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
    'ViCCT_large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth'
}


def make_save_dirs(loaded_cfg):
    """ Each run has its own directory structure, which is created here."""

    if not os.path.exists(loaded_cfg.SAVE_DIR):
        os.mkdir(loaded_cfg.SAVE_DIR)
        os.mkdir(loaded_cfg.PICS_DIR)
        os.mkdir(loaded_cfg.STATE_DICTS_DIR)
        os.mkdir(loaded_cfg.CODE_DIR)
        with open(os.path.join(cfg.SAVE_DIR, '__init__.py'), 'w') as f:  # For dynamic loading of config file
            pass
    else:
        print('save directory already exists!')


def main(cfg):
    """
    Main does the following
    1) Creates the save directories when starting training. Loads the directory paths when continue straining
        1.1) (new training only) Make a backup of some important files for archiving and reproducibility purposes.
    2) Sets seeds for reproducibility
    3) Makes the model
    4) Gets the function with which the datloaders can be obtained. Also loads the dataset specific settings.
    5) Makes the trainer object for training and calls train to train the model (also when fine-tuning)
    Loads the settings and model, then creates a trainer with which the model is trained."""

    if cfg.RESUME:  # Not fully tested yet
        module = importlib.import_module(cfg.RESUME_DIR.replace(os.sep, '.') + 'code.config')
        cfg = module.cfg
    else:  # Make a backup of some important files for archiving purposes.
        make_save_dirs(cfg)  # The folders to categorize the files
        copyfile('config.py', os.path.join(cfg.CODE_DIR, 'config.py'))
        copyfile('trainer_standard.py', os.path.join(cfg.CODE_DIR, 'trainer_standard.py'))
        copyfile('models/ViCCTModels.py', os.path.join(cfg.CODE_DIR, 'ViCCTModels.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, 'settings.py'),
                 os.path.join(cfg.CODE_DIR, 'settings.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, 'loading_data.py'),
                 os.path.join(cfg.CODE_DIR, 'loading_data.py'))
        copyfile(os.path.join('datasets', 'standard', cfg.DATASET, cfg.DATASET + '.py'),
                 os.path.join(cfg.CODE_DIR, cfg.DATASET + '.py'))

    # Seeds for reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    cudnn.benchmark = True  # Input to ViCCT is always of size batch_size x 224 x 224

    print(f"Creating model: {cfg.MODEL}")

    # Default settings from the original DeiT framework
    model = create_model(  # From the timm library. This function created the model specific architecture.
        cfg.MODEL,  # Which model to use (e.g. ViCCT tiny, ViCCT small, ViCCT base).
        init_path=model_mapping[cfg.MODEL],  # Where the pretrained weights of ImageNet are saved
        num_classes=1000,  # Not used. But must match pretrained model!
        drop_rate=0.,  # Dropout
        drop_path_rate=0.,  # Bamboozled by Facebook. This isn't drop_path_rate, but rather 'drop_connect'
        drop_block_rate=None,  # Drops our entire Transformer blocks I think? Not used for ViCCT.
    )

    model.cuda()  # CPU training not supported.

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)  # Print the number of trainable parameters of the model

    # Dynamically loads the dataloader and its settings as specified in the config file
    dataloader = importlib.import_module(f'datasets.{cfg.DATASET}.loading_data').loading_data
    cfg_data = importlib.import_module(f'datasets.{cfg.DATASET}.settings').cfg_data

    trainer = Trainer(model, dataloader, cfg, cfg_data)  # Make a trainer object
    trainer.train()  # Train the model


if __name__ == '__main__':
    main(cfg)
