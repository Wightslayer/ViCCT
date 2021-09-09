# See LICENCE for copyrights

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

import models.ViCCT_models  # Needed to register models for 'create_model'
import models.Swin_VICCT_models
from timm.models import create_model

from misc.model_zoo_links import model_mappings

import importlib

from trainer import Trainer
from config import cfg
from shutil import copyfile
import random


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
        copyfile('trainer.py', os.path.join(cfg.CODE_DIR, 'trainer.py'))
        copyfile('models/ViCCT_models.py', os.path.join(cfg.CODE_DIR, 'ViCCT_models.py'))
        copyfile(os.path.join('datasets', cfg.DATASET, 'settings.py'),
                 os.path.join(cfg.CODE_DIR, 'settings.py'))
        copyfile(os.path.join('datasets', cfg.DATASET, 'loading_data.py'),
                 os.path.join(cfg.CODE_DIR, 'loading_data.py'))
        copyfile(os.path.join('datasets', cfg.DATASET, cfg.DATASET + '.py'),
                 os.path.join(cfg.CODE_DIR, cfg.DATASET + '.py'))

    # Seeds for reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    cudnn.benchmark = True  # Input to ViCCT is always of size batch_size x 224 x 224

    print(f"Creating model: {cfg.MODEL}")

    # Default settings from the original DeiT framework
    model = create_model(  # From the timm library. This function created the model specific architecture.
        cfg.MODEL,
        init_path=model_mappings[cfg.MODEL],
        pretrained_cc=False,
        drop_rate=None if 'Swin' in cfg.MODEL else 0.,  # Dropout

        # Bamboozled by Facebook. This isn't drop_path_rate, but rather 'drop_connect'.
        # Not yet sure what it is for the Swin version
        drop_path_rate=None if 'Swin' in cfg.MODEL else 0.,
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
