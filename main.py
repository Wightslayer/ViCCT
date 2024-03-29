# See LICENCE for copyrights

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

import models.ViCCT_models  # Needed to register models for 'create_model'
import models.Swin_ViCCT_models
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


def backup_code(cfg):
    """ Backup all python files in this working directory. """

    base_dir = cfg.CODE_DIR
    for dirpath, dirnames, filenames in os.walk('.'):
        if '__pycache__' in dirpath or dirpath.startswith('.' + os.path.sep + '.') or dirpath.startswith('.' + os.path.sep + 'runs') or dirpath.startswith('.' + os.path.sep + 'notebooks') or dirpath.startswith('.' + os.path.sep + 'venv'):
            continue

        save_path = os.path.join(base_dir, dirpath)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for f_name in filenames:
            if f_name.endswith('.py'):
                s_path = os.path.join(dirpath, f_name)
                d_path = os.path.join(save_path, f_name)
                copyfile(s_path, d_path)


def get_model(cfg):
    """ Creates the ViCCT or Swin ViCCT model and initialises its weights. Returns that model."""

    print(f"Creating model: {cfg.MODEL}")

    if cfg.PRETRAINED:  # If we have a pretrained CROWD COUNTING model...
        init_weights_location = cfg.PRETRAINED_WEIGHTS  # Use the specified path
    else:
        init_weights_location = model_mappings[cfg.MODEL]  # Use the default location for the specific model

    if 'Swin' in cfg.MODEL:  # Swin as the base
        model = create_model(  # From the timm library. This function created the model specific architecture.
            cfg.MODEL,
            init_path=init_weights_location,
            pretrained_cc=cfg.PRETRAINED,
            drop_rate=None,  # Dropout
            drop_path_rate=None,
            drop_block_rate=None,  # Drops our entire Transformer blocks I think? Not used for ViCCT.
        )
    else:  # DeiT as the base
        model = create_model(  # From the timm library. This function created the model specific architecture.
            cfg.MODEL,
            init_path=init_weights_location,
            pretrained_cc=cfg.PRETRAINED,
            drop_rate=0.,  # Dropout

            # Bamboozled by Facebook. This isn't drop_path_rate, but rather 'drop_connect'.
            # Not yet sure what it is for the Swin version
            drop_path_rate=0.,
            drop_block_rate=None,  # Drops our entire Transformer blocks I think? Not used for ViCCT.
        )

    return model


def main(cfg):
    """
    Main does the following
    1) Create directories where stuff for this run is saved
    2) Makes a backup the code in this project
    2) Sets seeds for reproducibility
    3) Makes and initialises the model
    4) Gets the function with which the dataloaders can be obtained. Also loads the dataset specific settings.
    5) Makes the trainer object for training and calls train to train the model (also when fine-tuning)
    Loads the settings and model, then creates a trainer with which the model is trained."""

    make_save_dirs(cfg)  # These are the folders where we backup the code, save images and the model weights.
    backup_code(cfg)  # Make a backup of all code files

    # Seeds for reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    cudnn.benchmark = True  # Input to ViCCT is always of size batch_size x crop_size x crop_size

    model = get_model(cfg)  # Creates and initialises the model
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
