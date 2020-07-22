import os
import json
import glob
import SimpleITK as sitk
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dicomutil



jsonPath = r'D:\git\git_hub\tianchi\202006\lumbar_train51_annotation.json'
#jsonPath = r'D:\git\git_hub\tianchi\202006\lumbar_train150_annotation.json'

vertebras=[]
discs=[]
duogebiaoqian=[]
# studyUid,seriesUid,instanceUid,annotation
annotation_info = pd.DataFrame(columns=('studyUid','seriesUid','instanceUid','annotation'))
json_df = pd.read_json(jsonPath)
for idx in json_df.index:
    studyUid = json_df.loc[idx,"studyUid"]
    seriesUid = json_df.loc[idx,"data"][0]['seriesUid']
    instanceUid =  json_df.loc[idx,"data"][0]['instanceUid']
    annotation = json_df.loc[idx,"data"][0]['annotation']
    datapoints = annotation[0]['data']
    points =datapoints['point']
    spines=''
    coordxs=[]
    coordys=[]
    for p in points:
        coordx=p['coord'][0]
        coordxs.append(coordx)
        coordy = p['coord'][1]
        coordys.append(coordy)
        identification = p['tag']['identification']
        # T12-L1，没有-代表锥体
        classp = ''
        if(identification.find('-')==-1):
            vertebra = p['tag']['vertebra']
            vertebras.append(vertebra)
            if(vertebra.find(',')!=-1):#有多个值的时候，先只处理一个
                #vertebra=vertebra.split(',')[0]
                duogebiaoqian.append('vertebra:'+vertebra)
            if (vertebra == ''):  # 没有值，脏数据
                continue
            classp = 'vertebra_' + vertebra
        # elif(identification.find('T12-L1') !=-1):#T12-L1用于定位，单独处理下
        #     classp='T12-L1'

        else:
            disc = p['tag']['disc']
            discs.append(disc)
            if (disc.find(',')!=-1):#有多个值的时候，先只处理一个
                #disc = disc.split(',')[0]
                duogebiaoqian.append('disc:' + disc)
            if(disc==''):#没有值，脏数据
                continue
            classp = 'disc_' + disc

print('duogebiaoqian-size:',len(duogebiaoqian))
print('vertebras-size:',len(vertebras))
unique_data = np.unique(vertebras)
resdata = []
for ii in unique_data:
    print('{}有{}个'.format(ii,vertebras.count(ii)))

print('disc-size:',len(discs))
unique_data = np.unique(discs)
resdata = []
for ii in unique_data:
    print('{}有{}个'.format(ii,discs.count(ii)))


