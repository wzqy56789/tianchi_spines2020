import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pandas as pd
import glob
import dicomutil
import spines

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


#%matplotlib inline
start_time = time.time()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

weights_path=os.path.join(MODEL_DIR, "mask_rcnn_spines.h5")



config = spines.SpinesConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"



# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

classdict={1:"disc_v1",2:"disc_v2",3:"disc_v3",4:"disc_v4",5:"disc_v5",6:"vertebra_v1",7:"vertebra_v2",8:"T12_S1"}
tag_list = ['0020|000d','0020|000e','0008|0018','0008|103e','0018|1312']

#notcommit='commit'#提交
#notcommit='val'#验证测试集
notcommit='test'#测试单个
testpath = r'D:\git\git_hub\tianchi\202006\lumbar_testA50'
#testpath = r'D:\git\git_hub\tianchi\202006\lumbar_testA50\study249'
#testpath = r'D:\git\git_hub\tianchi\202006\lumbar_train51'
#testpath = 'D:/git/git_hub/tianchi/202006/sample_result_test.csv'
#dcm_paths = glob.glob(os.path.join(testpath,"**","**.dcm"))

predictjsompath=os.path.join(ROOT_DIR, "predictions_20200720.json")

if(notcommit =='test'):
    dcm_study_paths = [1]
elif (notcommit == 'val'):
    dcm_study_paths=[]
    csv_data = pd.read_csv(testpath, index_col='dcmpath')
    for path, row in csv_data.iterrows():
        dcm_study_paths.append(path)

else:
    dcm_study_paths = glob.glob(os.path.join(testpath, "**"))
    #dcm_study_paths=[testpath]
print('dcm_study_paths：',len(dcm_study_paths))
commitjson_final=[]
dcm_study_num=0
for dcm_study_path in dcm_study_paths:#处理每个study文件夹

    # if(dcm_study_num==5):#为了看效果，少跑几个
    #     break
    dcm_study_num=dcm_study_num+1
    print('处理第几个开始:', dcm_study_num)
    print('处理:', dcm_study_path)

    if (notcommit == 'test'):
        dcm_paths=['D:/git/git_hub/tianchi/202006/lumbar_testA50/study202/image13.dcm']
    elif(notcommit == 'val'):
        dcm_paths=[dcm_study_path]
    else:
        dcm_paths = glob.glob(os.path.join(dcm_study_path, "**.dcm"))

    commitjson={}
    commitjsonscore={}
    commitjson_t1=''
    i=0
    ##循环处理每个dicom
    for path in dcm_paths:

        studyUid,seriesUid,instanceUid,seriesDescription,rowcol = dicomutil.dicom_metainfo(path,tag_list)
        seriesDescription=seriesDescription.upper()
        if(seriesDescription.find('T2')==-1 and seriesDescription.find('IRFSE')==-1):#没找到T2，不处理。T2WI_SAG.#IRFSE 4000/90/123 5mmS
            #print('该文件不是T2矢状',path)
            continue
        if (seriesDescription.find('TRA') != -1):  #不处理TRA
            # print('该文件不是T2矢状',path)
            continue
            #C4-T2 FSE SAG  STIR, 15 slices.STIR=抑制脂肪序列,T2 IR-TSE
        if (seriesDescription.find('TRIM') != -1 or seriesDescription.find('IR') != -1 ):  # 不处理TRIM T2_FSE(T),t2_tirm_fs_sag
            # print('该文件不是T2矢状',path)
            continue
        if (seriesDescription.find('(T)') != -1 or seriesDescription.find('AXIAL') != -1):  # 不处理 T2_FSE(T)，T2_FSE_5mm(T)， C5-AXIAL T2 FSE PS, 12 slices
            # print('该文件不是T2矢状',path)
            continue
        #t2_tse_dixon_sag_320_F ，dixon 反应脂肪;:a6OT2 :a6OT2
        if (seriesDescription.find('DIXON') != -1 or seriesDescription.find('OT2') != -1):  # 不处理TRA
            # print('该文件不是T2矢状',path)
            continue
        #FST2_SAGc时一般有T2WI_SAGc
        if (seriesDescription.find('FST2') != -1 ):  # 不处理FST2
            # print('该文件不是T2矢状',path)
            continue
        if (rowcol.find('ROW') != -1):  # 不处理ROW
            # print('该文件不是T2矢状',path)
            continue
        i = i + 1
        #print('处理第几个：', i)
        #print('dcm_path:', path)
        #print('path:', path)
        image = dicomutil.dicom2array(path)
        #image = dicomutil.imgenhance(image)
        #image = dicomutil.histequ(image)
        #image = dicomutil.laplace_sharpen(image)

        # 为了简单处理，统一图片大小
        # image=cv2.resize(image,(512,512))
        # Extending the size of the image to be (h,w,1)
        # 将图片二维扩展成3维，加了一个维度
        image = image[..., np.newaxis]

        # Run object detection. verbose=0不显示日志
        results = model.detect([image], verbose=0)

        # Display results
        #ax = get_ax(1)
        r = results[0]
        if(notcommit=='test'):
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        classdict, r['scores'],
                                        title="Predictions",show_mask=False)
        T12_S1_index=-1
        T12_S1_flag = False
        for class_id in r['class_ids']:
            class_id_dsc = classdict[class_id]
            T12_S1_index=T12_S1_index+1
            if(class_id_dsc=='T12_S1'):
                #找到了'T12_S1'，可能有多个T12_S1_index，但按得分排序的，所以会找到第一个靠谱的
                #print('找到了T12_S1')
                T12_S1_flag=True
                break

        if(T12_S1_flag):    #print('找到了T12_S1')
            T12_S1_y1,T12_S1_x1,T12_S1_y2,T12_S1_x2=r['rois'][T12_S1_index]
            #T12_S1_y2更容易定位，T12_S1_y1可能不易定位，修正下T12_S1_y1
            #T12_S1_y1=max(T12_S1_y1,T12_S1_y2-200)
            #y方向5等分
            T12_S1_dengfen=abs(T12_S1_y2-T12_S1_y1)/5
        else:
            #没找到T12_S1，无法定位，暂时退出本study循环
            continue
        ##修正T12_S1_y1
        rois={}
        j1 = 0
        for coord in r['rois']:
            tag = r['class_ids'][j1]
            tagdisc = classdict[tag]
            #coordy = int((coord[0] + coord[2]) / 2)
            coordy1=coord[0]
            rois.update({coordy1:j1})
            j1=j1+1

        #reverse = True 降序
        sorted_x = sorted(rois,reverse=True)
        vertebra_num=0
        for coordy1 in sorted_x:
            j1 = rois[coordy1]
            tag = r['class_ids'][j1]
            tagdisc = classdict[tag]
            coord=r['rois'][j1]
            coordx1=coord[1]
            coordx2 = coord[3]
            coordy2 = coord[2]
            coordx=(coordx1+coordx2)/2
            coordy = (coordy1 + coordy2) / 2
            if (tagdisc.find('vertebra') != -1):
                if (coordx < T12_S1_x1 or coordx > T12_S1_x2):
                    # 该预测不在T12_S1范围内
                    continue
                if (coordy >T12_S1_y2 ):#主要是防止找到尾椎吧，第一根尾椎很像L5
                    # 该预测不在T12_S1范围内
                    continue
                vertebra_num=vertebra_num+1
                if(vertebra_num==1):

                    # t5_index=rois[sorted_x[0]]
                    # if (T12_S1_y2 - coordy2 > 50):  # 大太多再修正
                    T12_S1_y2 = min(T12_S1_y2, coordy2 + 15)  # #修正下底部坐标,找到了T5才算修正正确，否则可能修正错误
                    T12_S1_dengfen = abs(T12_S1_y2 - T12_S1_y1) / 5
                if(vertebra_num==5):#已经找到5个vertebra_num，就不继续找了。因为降序，从底部找起
                    #if(coordy1-T12_S1_y1>50):#大太多再修正
                    #T12_S1_y1=max(T12_S1_y1,coordy1-10)#第5个vertebra上面还有个disc，预留空间为vertebra的高度,暂时写10吧，上面的其实没用了，不用标注
                    T12_S1_y1 = coordy1 - 10#直接修正为L1的吧，无论T12_S1_y1是高于L1，还是低于L1
                    T12_S1_dengfen = abs(T12_S1_y2 - T12_S1_y1) / 5
                    break


        points=[]
        scores=0
        j=0
        for coord in r['rois']:
            # # 总体得分和只计算框内的得分会不会差不多。目前看只计算T12_S1框内的好一些。模型收敛后，总得分也许好
            #scores = scores + r['scores'][j]
            tag = r['class_ids'][j]
            tagdisc = classdict[tag]

            if(T12_S1_index == j):#T12_S1不作为结果写入json
                j=j+1
                continue

            point={
                "coord": [
                    0,
                    0
                ],
                "tag": {
                },
                "zIndex": 5
            }
            # coordy=(coord[0]+coord[2])/2
            # coordx =(coord[1] + coord[3]) / 2
            coordy = int((coord[0] + coord[2]) / 2)
            coordx = int((coord[1] + coord[3]) / 2)
            if(coordy<T12_S1_y1 or coordy>T12_S1_y2):
                #该预测不在T12_S1范围内
                j = j + 1
                continue
            if (coordx < T12_S1_x1 or coordx > T12_S1_x2):
                # 该预测不在T12_S1范围内
                j = j + 1
                continue

            # # 总体得分和只计算框内的得分会不会差不多。目前看只计算T12_S1框内的好一些。
            scores = scores + r['scores'][j]
            # 看看坐标点在五等分的那个里面。整除运算//
            tag_index_vertebra=int((coordy-T12_S1_y1)//T12_S1_dengfen+1)
            if(tag_index_vertebra>5):tag_index_vertebra=5
            if (tag_index_vertebra <1): tag_index_vertebra = 1
            #使用round圆整函数时，他的值是取最接近的整数，而且当两个整数一样接近时(x.5)，取偶数.比如L4-L5，下面取值接近4
            tag_index_disc = int(round((coordy - T12_S1_y1) / T12_S1_dengfen))

            point["coord"][0]=coordx
            point["coord"][1] = coordy
            #finalcoord={"coord":[coordy,coordx]}
            #分类部分


            tagdisc2=tagdisc.split('_')
            if (tagdisc.find('disc')!=-1):
                if(tag_index_disc==0):
                    finaltag={ "identification":"T12"+"-L"+str(int(tag_index_disc+1)),"disc": tagdisc2[1]}
                elif (tag_index_disc == 5):
                    finaltag = {"identification": "L" + str(tag_index_disc) + "-S1","disc": tagdisc2[1]}
                else:
                    finaltag = {"identification": "L" + str(tag_index_disc) + "-L" + str(int(tag_index_disc + 1)), "disc": tagdisc2[1]}
                #finaltag = {"disc": "v2"}
            elif (tagdisc.find('vertebra')!=-1):
                finaltag = {"identification":"L"+str(tag_index_vertebra),"vertebra": tagdisc2[1]}
                #finaltag = {"vertebra": "v1"}#先写死v1
                #"identification":"L5"
                #tag_index='L'+tag_index
            else:
                finaltag={}

            if(len(finaltag)>0):
                point["tag"] = finaltag
                #point["zIndex"] = 5#默认赋值个5
                #拼装point
                points.append(point)

            j=j+1


        jsontext = {"studyUid":studyUid,"version":"v0.1",
                     "data":[{
                         "seriesUid":seriesUid,"instanceUid":instanceUid,
                         "annotation":
                             [{"annotator": 54,
                               "data":
                                   {"point":points}
                               }]
                     }]
                    }
        #print('jsontext：',jsontext)
        #jsontext2 = json.dumps(jsontext)
        #print('jsontext2：',jsontext2)
        commitjson.update({path:jsontext})
        commitjsonscore.update({scores:path})
    #最终取分数scores最高的dicom,reverse=true则是倒序（从大到小），
    if(len(commitjsonscore)!=0):#也许一个都没预测到
        print('该study共预测了几个文件：',i)
        commitjsonscorekey=sorted(commitjsonscore.keys(),reverse=True)
        commitscore=commitjsonscorekey[0]
        commitpath=commitjsonscore.get(commitscore)
        print('commitpath：',commitpath)
        commitjson_t1=commitjson.get(commitpath)
    else:
        print('没有预测到一个文件:', dcm_study_path)

    #commitjson_t2 = json.dumps(commitjson_t1)

    #print('commitjson2：',commitjson_t2)
    if(commitjson_t1!=''):
        commitjson_final.append(commitjson_t1)

#commitjson_final_result = json.dumps(commitjson_final)
#print('commitjson_final_result：',commitjson_final_result)
#print('end')

with open(predictjsompath, 'w') as file:
    json.dump(commitjson_final, file,indent=4)
print('task completed, {} seconds used'.format(time.time() - start_time))