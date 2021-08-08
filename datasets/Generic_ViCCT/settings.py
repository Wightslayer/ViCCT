from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_BS = 10  # How many crops per training step. Each crop is taken from a separate image.
cfg_data.VAL_BS = 1  # Must be 1
cfg_data.N_WORKERS = 0  # Number of workers for the PyTorch dataloader

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





# =========================================================================== #
#                               TRAINING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TRAIN_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',  # TODO: change for linux chads
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\train_split.csv',
        'n_copies': 90,  # TODO: support this
        # 'percentage or something': None# TODO: Support this
    },
    {
        'dataset_name': 'ShanghaiTech_Part_A_10_percent',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A',  # TODO: change for linux chads
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A\\train_split.csv',
        'percent_of_split': 10
    },
    {
        'dataset_name': 'ShanghaiTech_Part_A_5_percent',
        'den_gen_key': 'SHT',
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A',  # TODO: change for linux chads
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A\\train_split.csv',
        'percent_of_split': 5
    }
]


# =========================================================================== #
#                              VALIDATION DATA SPLIT                          #
# =========================================================================== #
cfg_data.VAL_DATASETS = [
    # {
    #     'dataset_name': 'ShanghaiTech_Part_B',
    #     'den_gen_key': 'SHT',
    #     'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',
    #     'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\val_split.csv'
    # }
]

# =========================================================================== #
#                                TESTING DATA SPLIT                           #
# =========================================================================== #
cfg_data.TEST_DATASETS = [
    # {
    #     'dataset_name': 'ShanghaiTech_Part_A',
    #     'den_gen_key': 'SHT',
    #     'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A',
    #     'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_A\\test_split.pkl'
    # }
    # {
    #     'dataset_name': 'Muni',
    #     'den_gen_key': 'Municipality',
    #     'dataset_path': 'D:\\ThesisData\\Datasets\\Municipality\\Vondelpark_8_10May2020',  # TODO: change for linux chads
    #     'split_to_use_path': 'D:\\ThesisData\\Datasets\\Municipality\\Vondelpark_8_10May2020\\test_split.csv',
    #     # 'n_copies': 1,  # TODO: support this
    #     # 'percentage or something': None  # TODO: Support this
    #     'percent_of_split': 10,
    # }
]
