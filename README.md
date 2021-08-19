# Vision Crowd Counting Transformers (ViCCT) for Cross-Scene Crowd Counting

![Example of density map regression](./misc/example_images/ViCCT_architecture.jpg?raw=true "ViCCT architecture")

This repository contains the code for the ViCCT project. ViCCT is a fully transformer based crowd counter based on the [`ViT`](https://arxiv.org/abs/2010.11929) and [`DeiT`](https://arxiv.org/abs/2012.12877) architectures.


To perform crowd counting with ViCCT, we first split an image of any resolution into crops of size 224 x 224 or 384 x 384, depending on which ViCCT model is used. These crops are processed individually by ViCCT. Each crop is split into patches of size 16 x 16 and passed through the DeiT network. The output embeddings are resized with linear layers to be of shape 256. Each element of these vectors is considered a pixel of the density map. By concatinating all these resized embedding vectors, we obtain a crop of the final density map. These crops are combined to create the final density map. Overlap is resolved by taking the average pixel value of the overlapping pixels.


## Repository structure

* The structure of this repository is as follows:
  * The [`datasets`](/datasets) folder contains the ViCCT dataloader, including the code needed to load images from the disk and create the GT annotations. More information is provided in the README of the [`datasets`](/datasets) folder.
  * [`models`](/models) contains the code for the ViCCT models.
  * [`notebooks`](/notebooks) contains some usefull notebooks to create dataset splits (train/val/test), evaluate trained models, analyse datasets, and more.
  * [`config.py`](config.py) 


























