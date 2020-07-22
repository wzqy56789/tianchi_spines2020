"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import pandas as pd
import json
import dicomutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class SpinesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "spines"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 7锥体+椎间盘
    #NUM_CLASSES = 1 + 7  # background + 7锥体+椎间盘

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    #基本固定大小的，所以anchor的大小少弄几个。
    #RPN_ANCHOR_SCALES = (32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 90

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 30


class SpinesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_spines(self, count,height, width,datatype):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes，disc=椎间盘，vertebra=锥体
        self.add_class("spines", 1, "disc_v1")
        self.add_class("spines", 2, "disc_v2")
        self.add_class("spines", 3, "disc_v3")
        self.add_class("spines", 4, "disc_v4")
        self.add_class("spines", 5, "disc_v5")
        self.add_class("spines", 6, "vertebra_v1")
        self.add_class("spines", 7, "vertebra_v2")
        #self.add_class("spines", 8, "T12-L1")#T12-L1用来定位第一个椎间盘，从上到下
        self.add_class("spines", 8, "T12_S1")#T12-S1把全部的胸椎弄出来，定位

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().

        if(datatype=='train'):
            resultfile = 'D:/git/git_hub/tianchi/202006/sample_result.csv'
        elif (datatype == 'val'):
            resultfile = 'D:/git/git_hub/tianchi/202006/sample_result_val.csv'
        elif (datatype == 'test'):
            resultfile = 'D:/git/git_hub/tianchi/202006/sample_result_test.csv'
        else:#默认train
            resultfile = 'D:/git/git_hub/tianchi/202006/sample_result.csv'
        #resultfile = 'D:/tianchi/202006/sample_result.csv'

        csv_data = pd.read_csv(resultfile, index_col='dcmpath')
        print('csv_data len:', len(csv_data))
        i = 0
        for index, row in csv_data.iterrows():
            #print('第几行：', i)
            #print(index)#文件路径
            image = dicomutil.dicom2array(index)
            width=image.shape[1]
            height=image.shape[0]
            #print('--points---')
            #print(row['spines'])
            spines=row['spines']
            self.add_image("spines", image_id=i, path=index,
                           width=width, height=height,
                           spines=spines,imageenhance=0)
            #每个图片再来个数据增强的
            # self.add_image("spines", image_id=i, path=index,
            #                width=width, height=height,
            #                spines=spines,imageenhance=1)
            # i = i + 1
            # if(i>=count):#只获取固定数量的
            #     break


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path = info['path']
        #print('path:',path)
        image = dicomutil.dicom2array(path)
        imageenhance = info['imageenhance']
        if(imageenhance==1):#数据增强
            image = dicomutil.imgenhance(image)

        #为了简单处理，统一图片大小
        #image=cv2.resize(image,(512,512))
        # Extending the size of the image to be (h,w,1)
        #将图片二维扩展成3维，加了一个维度
        image = image[..., np.newaxis]
        # Resize
        # image, window, scale, padding, _ = utils.resize_image(
        #     image,
        #     min_dim=SpinesConfig.IMAGE_MIN_DIM,
        #     max_dim=SpinesConfig.IMAGE_MAX_DIM,
        #     mode=SpinesConfig.IMAGE_RESIZE_MODE)
        # #mask = utils.resize_mask(mask, scale, padding)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "spines":
            return info["spines"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        spines = info['spines']
        #spines_json = pd.read_json(spines)
        #spines_json = json.dumps(spines)
        spines_sp=spines.split(';')
        count = len(spines_sp)
        height=info['height']
        width=info['width']
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        #s = 20  # 长方形大小
        for i,spine in enumerate(spines_sp):
            sp=spine.split(':')
            #print('spine:',spine)
            #if(len(sp)):continue#如果最后一个，就退出。
            #print('x，y:',sp[0],sp[1])
            if(sp[2].find('-')==-1 and sp[2].find('_')==-1):
                #每张图片大小不一样，所以mask大小也不一样.
                #将width按模型尺寸512折算下
                x=sp[0]
                y = sp[1]
                sw=36/512*width #T12-L1，没有-代表锥体。那么高度更高一些
                sh=26/512*height
            elif(sp[2].find('T12')!=-1 and sp[2].find('S1')!=-1):#整个胸椎，mask较大
                T12_S1_x=sp[0]
                T12_S1_x_min=int(T12_S1_x.split('-')[0])
                T12_S1_x_max = int(T12_S1_x.split('-')[1])
                T12_S1_y = sp[1]
                T12_S1_y_min = int(T12_S1_y.split('-')[0])
                T12_S1_y_max = int(T12_S1_y.split('-')[1])
                x=(T12_S1_x_min+T12_S1_x_max)/2
                y = (T12_S1_y_min + T12_S1_y_max) / 2
                sw=36/512*width+abs(T12_S1_x_max-T12_S1_x_min)/2
                sh = 16 / 512 * height + abs(T12_S1_y_max - T12_S1_y_min)/2

            else:
                x = sp[0]
                y = sp[1]
                sw=36/512*width
                sh=26/512*height
            dims = (int(x),int(y),int(sw),int(sh))
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                   dims, 1)
        # #Handle occlusions
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(count - 2, -1, -1):#range(start, stop[, step])
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(
        #         occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s.split(':')[3]) for s in spines_sp])
        # load_image
        # image, window, scale, padding, _ = utils.resize_image(
        #     image,
        #     min_dim=SpinesConfig.IMAGE_MIN_DIM,
        #     max_dim=SpinesConfig.IMAGE_MAX_DIM,
        #     mode=SpinesConfig.IMAGE_RESIZE_MODE)
        # # mask = utils.resize_mask(mask, scale, padding)
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image,  dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, sw,sh = dims
        #矩阵的左上点坐标,矩阵的右下点坐标
        image = cv2.rectangle(image, (x - sw, y - sh),
                                  (x + sw, y + sh), color, -1)

        return image