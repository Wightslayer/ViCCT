from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_BS = 10  # How many crops per training step. Each crop is taken from a separate image.
cfg_data.VAL_BS = 1  # Must be 1
cfg_data.N_WORKERS = 4  # Number of workers for the PyTorch dataloader

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # From ImageNet
cfg_data.LABEL_FACTOR = 3000  # Scale each pixel in the GT density maps by this value

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


def oversample_generator(dataset_name, den_gen_key, dataset_path, split_to_use_path, n_copies):
    """ We might wish to add a dataset multiple times to, e.g., the training data. Instead of manually /c
        /v the dataset info below, this function generates n copies in a list.
        Append the return of this function to the cfg_data dataset list.
        (e.g. cfg_data.TRAIN_DATASETS = [ ... ] + oversample_generator(...))"""

    dupes = []
    for i in range(n_copies):
        dataset_dupe = {}
        dataset_dupe['dataset_name'] = dataset_name + '_copy_' + str(i + 1)
        dataset_dupe['den_gen_key'] = den_gen_key
        dataset_dupe['dataset_path'] = dataset_path
        dataset_dupe['split_to_use_path'] = split_to_use_path

        dupes.append(dataset_dupe)

    return dupes


# =========================================================================== #
#                               TRAINING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TRAIN_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\train_split.pkl'
    }
] + oversample_generator(
    dataset_name='ShanghaiTech_Part_B',
    den_gen_key='SHT',
    dataset_path='D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',
    split_to_use_path='D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\train_split.pkl',
    n_copies=10
)


# =========================================================================== #
#                              VALIDATION DATA SPLIT                          #
# =========================================================================== #
cfg_data.VAL_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\val_split.pkl'
    }
]

# =========================================================================== #
#                                TESTING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TEST_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_A',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A',
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A\\test_split.pkl'
    }
]
