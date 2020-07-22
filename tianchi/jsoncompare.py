import os
import json
import glob
import SimpleITK as sitk
import pandas as pd

tp=0
fp=0#坐标正确，但分类不正确
find=0
notfind=0
notfind_forpipei=0
total_study=0
total_points=0

ground_jsonPath = r'D:\git\git_hub\tianchi\202006\lumbar_train51_annotation.json'
predict_jsonPath = r'D:\git\git_hub\tianchi\202006\predictions_val.json'
ground_json2 = pd.read_json(ground_jsonPath)
predict_json2 = pd.read_json(predict_jsonPath)
for idx in predict_json2.index:
    total_study=total_study+1
    studyUid = predict_json2.loc[idx,"studyUid"]
    seriesUid = predict_json2.loc[idx,"data"][0]['seriesUid']
    instanceUid =  predict_json2.loc[idx,"data"][0]['instanceUid']
    annotation = predict_json2.loc[idx,"data"][0]['annotation']
    datapoints = annotation[0]['data']
    points =datapoints['point']
    total_points=total_points+len(points)
    for idx2 in ground_json2.index:
        studyUid2 = ground_json2.loc[idx2, "studyUid"]
        seriesUid2 = ground_json2.loc[idx2, "data"][0]['seriesUid']
        instanceUid2 = ground_json2.loc[idx2, "data"][0]['instanceUid']
        annotation2 = ground_json2.loc[idx2, "data"][0]['annotation']
        datapoints2 = annotation2[0]['data']
        points2 = datapoints2['point']
        if(studyUid==studyUid2 and seriesUid==seriesUid2 and instanceUid==instanceUid2):
            for p in points:
                coordx = p['coord'][0]
                coordy = p['coord'][1]
                identification = p['tag']['identification']
                findflag=False
                findflag_notpipei = False
                p_num=0
                for p2 in points2:
                    coordx2 = p2['coord'][0]
                    coordy2 = p2['coord'][1]
                    identification2 = p['tag']['identification']
                    if(identification==identification2 and abs(coordx-coordx2)<9 and abs(coordy-coordy2)<9):
                        p_num=p_num+1
                        if (identification.find('-') == -1):
                            vertebra = p['tag']['vertebra']
                            try:
                                vertebra2 = p2['tag']['vertebra']
                            except:
                                vertebra2 = ''
                            if(vertebra==vertebra2):
                                tp=tp+1
                                findflag=True

                                #print("找到第{} 个 ，坐标1-2是 {}、{}--{}、{}，vertebra分类是{}-{}".format(tp,coordx,coordy,coordx2,coordy2,vertebra,vertebra2))
                            else:
                                findflag_notpipei=True
                                # print(
                                #     "不匹配第 {} 个，坐标1-2是 {}、{}--{}、{}，vertebra分类是{}-{}".format(tp, coordx, coordy, coordx2,
                                #                                                             coordy2, vertebra,
                                #                                                             vertebra2))

                                fp=fp+1

                        else:
                            disc = p['tag']['disc']

                            try:
                                disc2 = p2['tag']['disc']
                            except:
                                disc2 = ''
                            if (disc == disc2):
                                findflag = True

                                tp = tp + 1
                                print(
                                    "找到第{}个  ，坐标1-2是 {}、{}--{}、{}，disc分类是{}".format(tp, coordx, coordy, coordx2,
                                                                                            coordy2, disc))
                            else:
                                findflag_notpipei = True
                                fp = fp + 1
                                # print(
                                #     "不匹配第{}个  ，坐标1-2是 {}、{}--{}、{}，disc分类是{}-{}".format(fp, coordx, coordy, coordx2,
                                #                                                      coordy2, disc,disc2))

                    # else:
                    #     notfind = notfind + 1


                if(p_num>1):
                    print('单个point是否匹配到，匹配的数量:',p_num)
                if(not findflag):notfind=notfind+1
                if ( findflag): find = find + 1
                if (findflag_notpipei): notfind_forpipei = notfind_forpipei + 1



print('总共predit_study几个：', total_study)
print('总共preditpoints几个：', total_points)
print('总共找到几个tp：', tp)
print('总共找到几个fp：', fp)
#print('分类错误的：', notfind_forpipei)
print('没有找到几个：', notfind)
#print('找到几个：', find)