import os
import json
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')

def dicom_metainfoall(dicm_path):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return reader.GetMetaData()

def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    result=[]
    for t in list_tag:
        try:
            r=reader.GetMetaData(t)
            result.append(r)
        except:
            #
            result.append('')
    #return [reader.GetMetaData(t) for t in list_tag]
    return result


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        #MONOCHROME1表示灰度范围从亮到暗,上升像素值,而MONOCHROME2范围从暗到亮,像素值上升.
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)
    img_x = img_x[0]
    return img_x


def imgenhance(image):
    #binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    #binary = cv2.blur(image, (3, 3))
    #binary=cv2.medianBlur(image, 3)
    #binary=cv2.bilateralFilter(image, 5, 21, 21)
    # 伽马变换
    average = np.average(image)
    if(average<25):
        gamma = 1.3
    else:
        gamma = 1.1
    #gamma = 0.9
    binary = np.power(image, gamma)
    #全局直方图均衡
    #binary = cv2.equalizeHist(image)
    # 自适应直方图均衡化
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # binary = clahe.apply(image)

    return binary

def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray

def laplace_sharpen(input_image, c=3):
    '''
    拉普拉斯锐化
    :param input_image: 输入图像
    :param c: 锐化系数
    :return: 输出图像
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本

    # 拉普拉斯滤波器
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ])

    input_image_cp = np.pad(input_image_cp, (1, 1), mode='constant', constant_values=0)  # 填充输入图像

    m, n = input_image_cp.shape  # 填充后的输入图像的尺寸

    output_image = np.copy(input_image_cp)  # 输出图像

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            R = np.sum(laplace_filter * input_image_cp[i - 1:i + 2, j - 1:j + 2])  # 拉普拉斯滤波器响应

            output_image[i, j] = input_image_cp[i, j] + c * R

    output_image = output_image[1:m - 1, 1:n - 1]  # 裁剪

    return output_image