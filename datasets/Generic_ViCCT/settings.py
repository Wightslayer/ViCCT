from easydict import EasyDict as edict
import os

cfg_data = edict()

cfg_data.TRAIN_BS = 10  # How many crops per training step. Each crop is taken from a separate image.
cfg_data.VAL_BS = 1  # Must be 1
cfg_data.N_WORKERS = 4  # Number of workers for the PyTorch dataloader

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # From ImageNet
cfg_data.LABEL_FACTOR = 3000  # Scale each pixel in the GT density maps by this value
cfg_data.USE_GAMMA_TRANSFORM = False
cfg_data.USE_GRAYSCALE_TRANSFORM = False

cfg_data.OVERLAP = 8        # For test images, how much overlap should crops have
cfg_data.IGNORE_BUFFER = 4  # When reconstructing the complete density map, how many pixels of edges of the crops
#                             should be ignored. No pixels are ignored at the edges of the complete density map.


# =========================================================================== #
#                      DEFINING THE TRAIN/VAL/TEST SPLITS                     #
# =========================================================================== #
# For each dataset in each training split, the following information must be provided:
# {
#     dataset_name: Name of the dataset. Only used for informative prints
#     den_gen_key: Each dataset has it's own particularities on how to generate the GT density map. This
#                  key specifies which for which dataset we are generating the density maps.
#                  Supported datasets with their keys are provided below
#     dataset_path: The location where the dataset is stored
#     split_to_use_path: The location where the relative paths of the images and annotations are stored
#     n_copies: (Optional, int) How many times to add this specific dataset.
#               Cannot be used together with 'percent_of_split'
#     percent_of_split: (Optional, int/float) When specified, this dataset will be copied dynamically such that it
#                       is approximately X% of the split. Cannot be used together with 'n_copies'
# }

# Supported datasets and their keys:
#       KEY                 DATASET
#  SHT':                ShanghaiTech A and B
#  UCF_QNRF_ECCV18':    UCF_QNRF
#  LSTN_FDST':          LSTN_FDST (i.e. Fudan ShanghaiTech)
#  JHU_CROWD_PlusPlus': JHU-Crowd++
#  NWPU_Crowd':         NWPU-Crowd
#  WorldExpo':          WorldExpo'10  (Only the pre-generated density maps)
#  Municipality':       Most/All municipality datasets





# =========================================================================== #
#                               TRAINING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TRAIN_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',
        'den_gen_key': 'SHT',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_B'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_B', 'train_and_val_split.csv'),
    },
    {
        'dataset_name': 'ShanghaiTech_Part_A',
        'den_gen_key': 'SHT',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_A'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_A', 'train_and_val_split.csv'),
    },
    {
        'dataset_name': 'LSTN_FDST',
        'den_gen_key': 'LSTN_FDST',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'LSTN_FDST_DATASET'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'LSTN_FDST_DATASET', 'train_and_val_split.csv'),
    },
    {
        'dataset_name': 'JHU-Crowd++',
        'den_gen_key': 'JHU_CROWD_PlusPlus',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'JHU-CROWD++'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'JHU-CROWD++', 'train_and_val_split.csv'),
    },
    {
        'dataset_name': 'NWPU-Crowd',
        'den_gen_key': 'NWPU_Crowd',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'NWPU-Crowd'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'NWPU-Crowd', 'train_and_val_split.csv'),
    },
    {
        'dataset_name': 'UCF-QNRF_ECCV18',
        'den_gen_key': 'UCF_QNRF_ECCV18',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'UCF-QNRF_ECCV18'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'UCF-QNRF_ECCV18', 'train_and_val_split.csv'),
    }
]


# =========================================================================== #
#                              VALIDATION DATA SPLIT                          #
# =========================================================================== #
cfg_data.VAL_DATASETS = [
    # Val split also in training data (train_and_val_split), thus not fair indicator of performance.
    {
        'dataset_name': 'ShanghaiTech_Part_B',
        'den_gen_key': 'SHT',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_B'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_B', 'val_split.csv'),
    }
]

# =========================================================================== #
#                                TESTING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TEST_DATASETS = [
    {
        'dataset_name': 'SHTB',
        'den_gen_key': 'SHT',
        'dataset_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'part_B'),
        'split_to_use_path': os.path.join('D:', 'ThesisData', 'Datasets', 'ShanghaiTech', 'Part_B', 'test_split.csv'),
    }
]
