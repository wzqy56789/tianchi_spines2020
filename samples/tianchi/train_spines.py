#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib
import matplotlib.pyplot as plt
import spines

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model_path = os.path.join(MODEL_DIR, "mask_rcnn_spines.h5")
# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


config = spines.SpinesConfig()
config.display()

# Training dataset
dataset_train = spines.SpinesDataset()
dataset_train.load_spines(200,config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],'train')
dataset_train.prepare()
#
# Validation dataset
dataset_val = spines.SpinesDataset()
dataset_val.load_spines(30, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1],'val')
dataset_val.prepare()

image_ids = np.random.choice(dataset_train.image_ids, 1)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    #visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

#训练了一半的话，可以继续加载训练
model.load_weights(model_path, by_name=True)

# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             layers='heads')
#每幅图像随机抽样，将对比度标准化为0.5至1.5倍。
# augmentation = imgaug.augmenters.Sometimes(0.5, [
#                     imgaug.augmenters.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
#                     imgaug.augmenters.ContrastNormalization((0.5, 1.5))
#                 ])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=1,
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually

model.keras_model.save_weights(model_path)


