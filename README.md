# tianchi_spines2020
Spark“数字人体”AI挑战赛 ——脊柱疾病智能诊断大赛baseline

1	简介：
我主要是基于mask-rcnn进行的，mask-rcnn可以进行物体识别，会输出物体的定位、物体分类、还有物体mask。

参与比赛的baseline，git地址：https://github.com/wzqy56789/tianchi_spines2020

2	代码介绍

1)	Mask目录：基于开源的https://github.com/matterport/Mask_RCNN。我修改了一些配置和方法，开源的主要用于RGB图片，而本次项目生成的灰度图。修改了一些方法适配于灰度图。
2)	Tianchi目录：本次项目的代码。主要文件如下：

	dataproduce.py: 因为训练的图片有8千多张，而给标注的只有151张，所以为了节省时间，我将标注的可以训练的151张图片信息暂时保存到csv文件中，训练时只从csv加载这151张图片的信息。

	spines.py：初始化图片和标注信息，方便mask rcnn进行训练，比较简单，只有几个方法。我是用标注给的x，y坐标，生成矩形mask。Bbox不用自己生成，mask会自动帮忙转成bbox的。为了定位，我多定义了一个分类T12_S1，把全部胸椎bbox找到，方便定位。

	train_spines.py：训练的文件，运行就可以训练了，训练30 Epoch后，loss可以降到1左右，然后就基本可以定位了。第一次训练时去掉代码model.load_weights(model_path, by_name=True)，后面训练时加上，代表从上次训练的结果进行训练。

	datacommit.py：生成提交的json文件。当然首先通过模型预测图片，然后通过bbox的坐标、分类生成json。在每个study文件夹中，通过seriesDescription只预测T2的图片，在所有T2图片中取得分最高的图片作为提交的结果。

3	待改进
1)	上面模型训练50 Epoch后，基本能取得0.35左右的成绩。但是对椎间盘的定位很少，不过对椎体定位准确，不确定训练更多epoch后，会取得更好的成绩。
2)	Mask，我采用矩形生成的方式可以改进，以便对椎间盘定位更准确。
3)	分类，我没有考虑多个分类的，比如同时是v3、v5的情况。
4)	还没有用脊椎轴状图。
