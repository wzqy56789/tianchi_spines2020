import os
import json
import glob
import SimpleITK as sitk
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import dicomutil


trainPath = r'D:\git\git_hub\tianchi\202006\lumbar_train51'
jsonPath = r'D:\git\git_hub\tianchi\202006\lumbar_train51_annotation.json'
#trainPath = r'D:\git\git_hub\tianchi\202006\lumbar_train150'
#jsonPath = r'D:\git\git_hub\tianchi\202006\lumbar_train150_annotation.json'
# trainPath = r'D:\tianchi\202006\lumbar_train150'
# jsonPath = r'D:\tianchi\202006\lumbar_train150_annotation.json'
#resultfile = 'D:/git/git_hub/tianchi/202006/sample_result.csv'
resultfile = 'D:/git/git_hub/tianchi/202006/sample_result_val.csv'
#resultfile = 'D:/tianchi/202006/sample_result_val.csv'

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
            if(vertebra.find(',')!=-1):#有多个值的时候，先只处理一个
                vertebra=vertebra.split(',')[0]
            if (vertebra == ''):  # 没有值，脏数据
                continue
            classp = 'vertebra_' + vertebra
        # elif(identification.find('T12-L1') !=-1):#T12-L1用于定位，单独处理下
        #     classp='T12-L1'

        else:
            disc = p['tag']['disc']
            if (disc.find(',')!=-1):#有多个值的时候，先只处理一个
                disc = disc.split(',')[0]
            if(disc==''):#没有值，脏数据
                continue
            classp = 'disc_' + disc

        spines=spines+str(coordx)+':'+str(coordy)+':'+identification+':'+classp+';'
    #拼装整个胸椎
    #coordxs=sorted(coordxs,reverse=True)#reverse=true则是倒序（从大到小），
    #coordys = sorted(coordys, reverse=True)  # reverse=true则是倒序（从大到小），
    T12_S1_x=str(min(coordxs))+'-'+str(max(coordxs))
    T12_S1_y= str(min(coordys)) + '-' + str(max(coordys))
    spines = spines +T12_S1_x + ':' + T12_S1_y+ ':' + 'T12_S1' + ':' + 'T12_S1' + ';'
    spines=spines[:-1]#去掉最后一个分号
    row = pd.Series({'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid,'spines':spines})
    annotation_info = annotation_info.append(row,ignore_index=True)

print('annotation_info数量：',len(annotation_info))

dcm_paths = glob.glob(os.path.join(trainPath,"**","**.dcm"))
#dcm_paths = glob.glob(os.path.join(trainPath,"**.dcm"))
print('dcm_paths数量：',len(dcm_paths))
# 'studyUid','seriesUid','instanceUid'
tag_list = ['0020|000d','0020|000e','0008|0018']
dcm_info = pd.DataFrame(columns=('dcmpath','studyUid','seriesUid','instanceUid'))
i=0;
for dcm_path in dcm_paths:
    try:
        i = i + 1
        print('处理第几个：', i)
        print('dcm_path:',dcm_path)
        studyUid,seriesUid,instanceUid = dicomutil.dicom_metainfo(dcm_path,tag_list)
        row = pd.Series({'dcmpath':dcm_path,'studyUid':studyUid,'seriesUid':seriesUid,'instanceUid':instanceUid })
        dcm_info = dcm_info.append(row,ignore_index=True)
    except:
        continue
#merge：默认返回交集inner: 交集
result = pd.merge(annotation_info,dcm_info,on=['studyUid','seriesUid','instanceUid'])
result = result.set_index('dcmpath')['spines']

print('----result---------')
print('len:',len(result))
print(result)
"""
1.r(Read，读取)：对文件而言，具有读取文件内容的权限；对目录来说，
  具有浏览目 录的权限。 

2.w(Write,写入)：对文件而言，具有新增、修改文件内容的权限；对目
  录来说，具有删除、移动目录内文件的权限。

3.x(eXecute，执行)：对文件而言，具有执行文件的权限；对目录了来说
  该用户具有进入目录的权限。
"""

result.to_csv(resultfile, index=True, header=True)