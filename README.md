# Bus-Net
Hello and welcome to BUS-Set, a collection of several BUS datasets in the form of a benchmark.

DATASETS

The following links are the locations of the datasets used within the study:

OASBUD : https://zenodo.org/record/545928#.Y_TIs4DP20n

BUSI: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

UDIAT : http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php

RODTOOK : http://www.onlinemedicalimages.com/index.php/en/81-site-info/73-introduction

Plus BUSIS : http://cvprip.cs.usu.edu/busbench/  but this was not used during the study


The OASBUD and BUSI are simple downloads, for RODTOOK you will have to navigator the website and download the images. 
Then for UDIAT a lisensing agreement will need to be signed. 

The only dataset that will need prep is RODTOOK, which will require all the surrounding annotations removed through cropping.

MODELS

For Benchmarking BUS-Set 9 models were used; Mask-RCNN, Deeplab v3+, U-Net, Sk-U-Net, Att-Dense-U-Net, Att-U-Net, Swin-U-Net, Trans-U-Net.

We have provided code for all the models that we impletemented ourselfs which include: Deeplab v3+, U-Net, Sk-U-Net, Att-Dense-U-Net, Att-U-Net. 
These are avaiable in "scrpts_to_run_each_model"

For the remaining models, they were impletement from the following:

Mask-RCNN : https://github.com/matterport/Mask_RCNN
Swin-U-Net : https://github.com/HuCaoFighting/Swin-Unet
Trans-U-Net : https://github.com/Beckschen/TransUNet

Swin-U-Net and Trans-U-Net are in pytorch and the remining in tensorflow

All Pretrained model wieghts are available at request, please email cot12@aber.ac.uk (due to file sizes)

But all of our best prediction masks can be found here:https://drive.google.com/file/d/1s9xr1UNmLwd1L8Vx-tX5t9XnDclh5z7c/view?usp=share_link
