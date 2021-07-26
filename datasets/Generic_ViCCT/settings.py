from easydict import EasyDict as edict

cfg_data = edict()

cfg_data.TRAIN_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',  # Only for informative prints
        'den_gen_key': 'SHT',  # Which get_gt to use in density_generators.py
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',  # Where to find dataset
        # Path to pickle file containing all the images and gts to use
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\train_split.pkl'
    }
]

cfg_data.VAL_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',  # Only for informative prints
        'den_gen_key': 'SHT',  # Which get_gt to use in density_generators.py
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',  # Where to find dataset
        # Path to pickle file containing all the images and gts to use
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\val_split.pkl'
    }
]

cfg_data.TEST_DATASETS = [
    {
        'dataset_name': 'ShanghaiTech_Part_B',  # Only for informative prints
        'den_gen_key': 'SHT',  # Which get_gt to use in density_generators.py
        'dataset_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B',  # Where to find dataset
        # Path to pickle file containing all the images and gts to use
        'split_to_use_path': 'D:\\ThesisData\\Datasets\\ShanghaiTech\\part_B\\test_split.pkl'
    }
]

cfg_data.TRAIN_BS = 10
cfg_data.VAL_BS = 1
cfg_data.N_WORKERS = 4

cfg_data.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cfg_data.LABEL_FACTOR = 3000

cfg_data.OVERLAP = 8  # For test images, how much overlap should crops have
cfg_data.IGNORE_BUFFER = 4  # When reconstructing the complete density map, how many pixels of edges should be ignored
                             # No pixels are ignored at the edges of the density map.

