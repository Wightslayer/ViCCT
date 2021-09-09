from easydict import EasyDict as edict
import time
import os
import math

cfg = edict()

# Valid_model_names = [
#     ViCCT_tiny, ViCCT_small, ViCCT_base, ViCCT_large
# ]


cfg.SEED = 42  # Seed for reproducibility.

# Select the model variant to use and which dataset to train it on.
# The model must be selected from above (Valid_model_names) AND the dataset name MUST match the name in the dataset
# folder. E.g. SHTB_DeiT to load SHTB for DeiT. Just SHTB does not work!
cfg.MODEL = 'ViCCT_tiny'
cfg.DATASET = 'Generic_ViCCT'

# Training parameters
cfg.LR = 1e-4  # LR for meta learning in meta learning. Standard LR for standard learning
cfg.LR_GAMMA = math.sqrt(0.1)  # Scale LR by this at each step in LR_STEP_EPCH
cfg.LR_STEP_EPOCHS = [100, 500, 900]  # Make one step with the learning rate scheduler at these epochs
cfg.WEIGHT_DECAY = 1e-5
cfg.GRAD_CLIP_NORM = 1.  # Attempt to make meta-learning more stable. Set to None for no clipping


cfg.MAX_EPOCH = 1300  # Train for this many epochs
cfg.EVAL_EVERY = 10  # Eval the model on the evaluation set every 'EVAL_EVERY' epochs
cfg.SAVE_EVERY_N_EVALS = 10  # Every Nth evaluation, save model regardless of performance

# Don't touch this one. Specifies at which epochs to save the model.
cfg.SAVE_EVERY = cfg.SAVE_EVERY_N_EVALS * cfg.EVAL_EVERY

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


# ===================================================================================== #
#                                  RESUME TRAINING                                      #
# ===================================================================================== #
#  We can resume training of a model. Can be used to fine-tune a model.
cfg.RESUME = False  # Whether to resume training or not
# cfg.RESUME_DIR = os.path.join('runs', '02-03_18-43')  # Alther the date-time to where the save dir of the run is.
# cfg.RESUME_STATE = 'save_state_ep_200_new_best_MAE_2.002.pth'  # With which model to continue training

# Automatically makes the complete path from where to resume training. Don't alter this one.
# cfg.RESUME_PATH = os.path.join('runs', cfg.RESUME_DIR, 'state_dicts', cfg.RESUME_STATE)



