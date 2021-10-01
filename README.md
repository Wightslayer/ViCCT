# Vision Crowd Counting Transformers (ViCCT) for Cross-Scene Crowd Counting

![Example of density map regression](./misc/example_images/ViCCT_architecture.jpg?raw=true "ViCCT architecture")

This repository contains the code for the ViCCT project. ViCCT is a fully transformer based crowd counter based on the [`ViT`](https://arxiv.org/abs/2010.11929) and [`DeiT`](https://arxiv.org/abs/2012.12877) architectures.


To perform crowd counting with ViCCT, we first split an image of any resolution into crops of size 224 x 224 or 384 x 384, depending on which ViCCT model is used. These crops are processed individually by ViCCT. Each crop is split into patches of size 16 x 16 and passed through the DeiT network. The output embeddings are resized with linear layers to be of shape 256. Each element of these vectors is considered a pixel of the density map. By concatinating all these resized embedding vectors, we obtain a crop of the final density map. These crops are combined to create the final density map. Overlap is resolved by taking the average pixel value of the overlapping pixels.


## Repository structure

* The structure of this repository is as follows:
  * The [`datasets`](/datasets) folder contains the ViCCT dataloader, including the code needed to load images from the disk and create the GT annotations. More information is provided in the README of the [`datasets`](/datasets) folder.
  * [`models`](/models) contains the code for the ViCCT models.
  * [`notebooks`](/notebooks) contains some usefull notebooks to create dataset splits (train/val/test), evaluate trained models, analyse datasets, and more.
  * [`config.py`](config.py) contains the settings for a particular training run. Note that settings specific to the datasets and dataloader are specified in [`datasets/Generic_ViCCT`](/datasets/Generic_ViCCT).
  * [`trainer.py`](trainer.py) contains the trainer class, based on the one from the [`C^3-Framework`](https://arxiv.org/abs/1907.02724), that trains a given ViCCT model.
  * Starting a training run is done with [`main.py`](main.py)


## Repository Installation
Clone this repository to a directory of your own choosing on your computer:
- git clone git@github.com:Wightslayer/ViCCT.git).

### Python Installation
Install Python 3.8, and the pip package manager for Python 3.8:
- `sudo apt install python3.8 python3.8-pip`

### Environment Installation
We set up a virtual Python environment, and install all necessary Python packages in this environment:
- Navigate your terminal to the folder where you cloned the ViCCT repository.
- `python3.8 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`


## Repository usage

### Training a Generic ViCCT model
#### 1. Download the following public datasets
- ShanghaiTech - Part A: https://www.kaggle.com/tthien/shanghaitech
- ShanghaiTech - Part B: https://www.kaggle.com/tthien/shanghaitech
- LSTN FDST: https://github.com/sweetyy83/Lstn_fdst_dataset
- JHU-Crowd++: http://www.crowd-counting.com/
- NWPU-Crowd: https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EsubMp48wwJDiH0YlT82NYYBmY9L0s-FprrBcoaAJkI1rw?e=e2JLgD
- UCF-Q NRF: https://www.crcv.ucf.edu/data/ucf-qnrf/
- We suggest creating 'datasets' folder in which all these downloaded datasets can be placed.


#### 2. Make train/val/test splits for the downloaded datasets
- Navigate to the folder notebooks/Make_train_val_test_splits in your terminal.
- Start Juypter Notebook inside your activated virtual environment:
  - Navigate your terminal to the folder where you cloned the ViCCT repository.
  - Activate the virtual environment: `source venv/bin/activate`
  - Start Juyter Notebook: `jupyter notebook`
- For each of the downloaded datasets in step 1, open the corresponding notebook.
- Modify the 'base_path' in the third cell of each notebook, to make it point to the folder containing the corresponding dataset. Then run all the cells in each notebook. This should create multiple .csv files in the dataset folder, linking to the images/gt files, and representing train/val/test splits for the dataset. (the file structure of the dataset-downloads might differ from what the notebooks assume when using different versions of the datasets. If this is the case, please update the notebooks to match the dataset structure).


#### 3. Run main.py to train the model
- Note: Traning to convergence can take a long time (we have been training some models for up to 2 weeks for the largest ViCCT versions using our suggested ViCCT config settings, and using all public datasets mentioned in step 1, and using a 2080 Ti).
- To check the training results, you can use tensorboard:
  - Open a new terminal in your local ViCCT repository folder.
  - Activate the virtual environment: `source venv/bin/activate`
  - Start Tensorboard: `tensorboard --logdir=runs`
  - Open Tensorboard in your browser to check the run results (usually at: http://localhost:6006/).
- The resulting network weights will be saved in the `runs` folder. This folder contains all runs and their results. The resulting network weights of a run will also be stored in .pth files within the folder of a run (be default, a .pth file is saved every 100 epochs). These network weights can be used to initiate a network in order to make crowd counting predictions with the network. NB: when using these weights, make sure that the same network type is used as during training; e.g. ViCCT_tiny, ViCCT_large, etc.


### Using the Generic ViCCT model
To use the model, make sure that Jupyter Notebook is running (using your virtual environment):
  - Activate the virtual environment: `source venv/bin/activate`
  - Start Juyter Notebook: `jupyter notebook`

Open the notebook "notebooks/Make image prediction.ipynb". In the third code-cell the 'weights_path' can be set to a .pth file resulting from one of your own training runs, or to one of the .pth files in the 'models/trained_models' folder (make sure the 'model_name' matches the network type used during training). The 'image_path' can be changed to point to a local image (e.g. .jpg or .png files). Next, all cells in the notebook can be ran in order to get a density map and crowd counting prediction for the image located on the image_path. You can change the 'image_path' to point to another image and run the entire notebook again to get another prediction.



# Acknowledgements

The code in this repository is heavily inspired by, and uses parts of, the Crowd Counting Code Framework ([`C^3-Framework`](https://github.com/gjy3035/C-3-Framework)). I also use and extend the code from the DeiT [`repository`](https://github.com/facebookresearch/deit) repository for the ViCCT models.


Important papers for this repository:
 - C^3-Framework [`paper`](https://arxiv.org/abs/1907.02724)
 - ViT [`paper`](https://arxiv.org/abs/2010.11929)
 - DeiT [`paper`](https://arxiv.org/abs/2012.12877)






















