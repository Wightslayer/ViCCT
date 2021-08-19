## Folder content
The [`Generic_ViCCT`](/Generic_ViCCT) directory contains the dataloader code and the settings for the dataloader. 

[`data_retriever_and_generator.py`](data_retriever_and_generator.py) contains code to load the image and GT annotation from the disk. It also produces the GT density map from the annotations. Since each dataset has a different style to save the annotations, each supported dataset has its own function to retrieve and make the GT density map.

[`dataset_utils.py`](dataset_utils.py) contains two important functions 'img_equal_split' and 'img_equal_unsplit'. These are the functions that split an image of any size into crops, and to combine the crops. Splitting the crops is done such that all crops have equal overlap to their adjecent crops. The amount of desired overlap is set with the 'overlap' parameter in [`settings.py`](/datasets/Generic_ViCCT/settings.py), and 'img_equal_split' ensures that there is at least that many pixels of overlap. For 'img_equal_unsplit', one could also specify how many pixels at the edges of crops to ignore with 'ignore_buffer', when reconstructing the complete density maps. This might be useful if you suspect that ViCCT is not able to discern human from background at the edges of image crops. Edges of crops that are at the border of an image are never ignored. Please note that 'overlap' must be at least twice as large as 'ignore_buffer'


[`transforms.py`](transforms.py) contains some image and GT transformations for data augmentation.

## Configuring your dataloader
In the [`Generic_ViCCT`](/Generic_ViCCT) folder, there are two files that must be modified for each run: 'setting.py' and 'loading_data.py'. In settings.py, the settings for the dataloader are set, as well as the specific datasets to use. In 'loading_data.py', one can specify which data-augmentations to use.


