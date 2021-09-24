from easydict import EasyDict as edict
import time
import os
import math

cfg = edict()

# Standard ViCCT models: ViCCT_tiny, ViCCT_small, ViCCT_base, ViCCT_large
# Swin ViCCT models: Swin_ViCCT_tiny, Swin_ViCCT_small, Swin_ViCCT_base, Swin_ViCCT_large, Swin_ViCCT_large_22k


cfg.SEED = 42  # Seed for reproducibility.

# Select the model variant to use and which dataset to train it on.
# The model must be selected from above AND the dataset name MUST match the name in the dataset
# folder. E.g. SHTB_DeiT to load SHTB for DeiT. Just SHTB does not work!
cfg.MODEL = 'ViCCT_small'
cfg.DATASET = 'Generic_ViCCT'

# PRETRAINED is for if we want to load a pretrained CROWD COUNTING MODEL.
# PRETRAINED_WEIGHTS is the path where these weights are stored.
cfg.PRETRAINED = True
# cfg.PRETRAINED_WEIGHTS = 'D:\\OneDrive\\OneDrive - UvA\\ThesisData\\trained_models\\SWIN generic\\save_state_ep_400.pth'
cfg.PRETRAINED_WEIGHTS = 'D:\\OneDrive\\OneDrive - UvA\\ThesisData\\trained_models\\ViCCT small TL SHTB\\save_state_ep_840_new_best_MAE_8.063.pth'

# Training parameters
cfg.LR = 1e-4  # LR for meta learning in meta learning. Standard LR for standard learning
cfg.LR_GAMMA = math.sqrt(0.1)  # Scale LR by this at each step in LR_STEP_EPCH
cfg.LR_STEP_EPOCHS = [100, 500, 900]  # Make one step with the learning rate scheduler at these epochs
cfg.WEIGHT_DECAY = 1e-5

cfg.MAX_EPOCH = 1300  # Train for this many epochs
cfg.EVAL_EVERY = 10  # Eval the model on the evaluation set every 'EVAL_EVERY' epochs
cfg.SAVE_EVERY = 100  # Save the model weights every this many epochs, regardless of performance

# Used to specify how many example predictions to save of evaluation.
# I.e. with every evaluation, save this many predictions.
cfg.SAVE_NUM_EVAL_EXAMPLES = 10


# ===================================================================================== #
#                                 SAVE DIRECTORIES                                      #
# ===================================================================================== #
# We make backups of some files. We also save the model state during training. Etc.
# These directories specify where to save all those files.
# They are created dynamically based on the current date and time
runs_dir = 'runs'  # Base directory (shared amongst all runs)
if not os.path.exists(runs_dir):
    os.mkdir(runs_dir)
    with open(os.path.join(runs_dir, '__init__.py'), 'w') as f:  # For dynamic loading of config file
        pass

# The following directories are created in main, but the paths are defined here.
cfg.SAVE_DIR = os.path.join(runs_dir, time.strftime("%m-%d_%H-%M", time.localtime()))  # Run specific save dir
cfg.PICS_DIR = os.path.join(cfg.SAVE_DIR, 'pics')  # Here we save pictures
cfg.STATE_DICTS_DIR = os.path.join(cfg.SAVE_DIR, 'state_dicts')  # Here we save the model state, etc.
cfg.CODE_DIR = os.path.join(cfg.SAVE_DIR, 'code')  # Here we save a backup of some files
