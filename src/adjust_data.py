"""
Author: Cong
Date: 2018-10-08
"""

# import the necessary packages
import numpy as np
import argparse
import os
import logging
import shutil

logging.basicConfig(level=logging.DEBUG)

label_dict = {"正常":0, "不导电":1, "擦花":2, "横条压凹":3, "桔皮":4, "漏底":5, "碰伤":6, "起坑":7, "凸粉":8,
        "涂层开裂":9, "脏点":10, "其他":11}

imageNum_dict = {} # 每一个分类的图片数量

def args_parse():
# 参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--dataPath", required=True,
        help="the path to data those need be adjusted")
    ap.add_argument("-n", "--number", required=True,
        help="the number of images of per classification")
    ap.add_argument("-o", "--outPath", required=True,
        help="the path of new data output")
    args = vars(ap.parse_args())    
    return args

def labelFromImageName(imageName):
    label_name = imageName.split(os.path.sep)[-1].split(".")[0].split("2018")[0]
    try:
        label = int(label_dict[label_name])
    except:
        label = 11
    return label_name, label # label_name是分类的中文名， label是对应的数字

    
def adjustData(args):
# 调整数据 
    data_dir = args["dataPath"]
    classPerNum = int(args["number"]) # 每个分类需要多少张图片，这里默认0分类，即正常分类图片为number的两倍
    outputPath = args["outPath"]
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    print("[INFO] loading data...")
    image_list = os.listdir(data_dir)
    for image in image_list:
        label_name, label = labelFromImageName(image)
        logging.debug("image name is %s, label name is %s" % (label_name, label))
        imageNum_dict[label] = imageNum_dict.get(label, 0) + 1
    needI = {} # 每个分类还需要多少图片
    for key in imageNum_dict.keys():

        if imageNum_dict.get(key, 0) >= classPerNum: 
            needI[key] = 1
            continue
        else: 
            needI[key] = classPerNum//imageNum_dict.get(key, 1) + 1
    needI = sorted(needI.items(), key=lambda x: x[0])
    needI = dict(needI)
    needI[0] = needI.get(0,1) + 1
    logging.debug("image of every classification need: %s" % (needI))

    newImageNameI = {}
    newImageNum_dict = {}
    classPerNum_dict = {}
    count = 0
    for key in imageNum_dict.keys():
        classPerNum_dict[key] = classPerNum_dict.get(key, classPerNum)
    classPerNum_dict[0] = 2 * classPerNum_dict.get(key, classPerNum) # 正常分类的两倍classPerNum数量
    for image in image_list:
        if (image.split(os.path.sep)[-1].split(".")[1]) != "jpg": 
            continue
        label_name, label = labelFromImageName(image)
        if(newImageNum_dict.get(label,0) < classPerNum_dict[label]):
            for i in range(needI.get(label, 1)):
                if(newImageNum_dict.get(label,0) >= classPerNum_dict[label]): continue
                newImageNum_dict[label] = newImageNum_dict.get(label,0) + 1
                oldname= data_dir + "/" + image
                newname= outputPath + "/" + label_name + "_" + str(label) + "_" + str(newImageNameI.get(label, 0)) + "." + "jpg"
                newImageNameI[label] = newImageNameI.get(label, 0) + 1

                shutil.copyfile(oldname,newname)
                count += 1

    logging.debug("The number of new data of label %s", newImageNum_dict)
    logging.debug("Done, total images number: %s", count)

# python adjust_data.py --dataPath /Users/cong/tianchi2018/data_all2 --number 200 --outPath /Users/cong/tianchi2018/data_new
if __name__ == '__main__':
    args = args_parse()
    adjustData(args)
