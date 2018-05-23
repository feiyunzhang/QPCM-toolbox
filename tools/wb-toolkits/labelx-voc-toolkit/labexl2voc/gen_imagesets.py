# -*- coding:utf-8 -*-

import random
import argparse
import os
from os import listdir
from os.path import join, isfile
import time
import json
import xml_helper
import utils
import numpy as np
'''
设置trainval和test数据集包含的图片
'''
def gen_imagesets(vocpath=None):
    # ImageSets文件夹
    _IMAGE_SETS_PATH = join(vocpath, 'ImageSets')
    _MAin_PATH = join(vocpath, 'ImageSets/Main')
    _XML_FILE_PATH = join(vocpath, 'Annotations')

    # 创建ImageSets数据集
    if os.path.exists(_IMAGE_SETS_PATH):
        print('ImageSets dir is already exists')
        if os.path.exists(_MAin_PATH):
            print('Main dir is already in ImageSets')
        else:
            os.makedirs(_MAin_PATH)
    else:
        os.makedirs(_IMAGE_SETS_PATH)
        os.makedirs(_MAin_PATH)

    # 遍历XML文件夹
    xml_list = [x for x in listdir(_XML_FILE_PATH) if isfile(join(_XML_FILE_PATH, x)) and not x[0] == '.']
    random.shuffle(xml_list)
    xml_numbers = len(xml_list)
    test_percent, train_percent, val_percent = 0.07, 0.77, 0.16
    test_list = xml_list[:int(xml_numbers*test_percent)]
    train_list = xml_list[int(xml_numbers * test_percent):int(xml_numbers * (test_percent+train_percent))]
    val_list = xml_list[int(xml_numbers * (test_percent+train_percent)):]
    trainval_list = train_list + val_list

    r = '\n'.join([xml[:xml.rfind('.')] for xml in test_list])
    with open(os.path.join(_MAin_PATH, 'test.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in train_list])
    with open(os.path.join(_MAin_PATH, 'train.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in val_list])
    with open(os.path.join(_MAin_PATH, 'val.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in trainval_list])
    with open(os.path.join(_MAin_PATH, 'trainval.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')
    # write readme in vocpath
    all_readme_dict = dict()
    readme_dict = dict()
    readme_dict['date'] = utils.getTimeFlag()
    readme_dict['dataInfo'] = [vocpath.split('/')[-1]]
    readme_dict['author'] = "Ben"
    readme_dict['total_num'] = xml_numbers
    readme_dict['trainval_num'] = len(trainval_list)
    readme_dict['test_num'] = len(test_list)
    # statistic bbox info
    all_readme_dict['imageInfo'] = readme_dict
    bboxInfo_dict = dict()
    bboxInfo_dict['trainval_bbox_info'] = statisticBboxInfo(
        imagelistFile=os.path.join(_MAin_PATH, 'trainval.txt'), xmlFileBasePath=_XML_FILE_PATH)
    bboxInfo_dict['test_bbox_info'] = statisticBboxInfo(
        imagelistFile=os.path.join(_MAin_PATH, 'test.txt'), xmlFileBasePath=_XML_FILE_PATH)
    all_readme_dict['bboxInfo'] = bboxInfo_dict
    readme_file = os.path.join(vocpath,'readme.txt')
    with open(readme_file,'w') as f:
        json.dump(all_readme_dict, f, indent=4)

def statisticBboxInfo(imagelistFile=None,xmlFileBasePath=None,printFlag=True):
    """
      imagelistFile is file , per line is a image(xml) file 
        not include jpg or xml 
    """
    line_count = 0
    label_count_dict=dict()
    with open(imagelistFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line :
                continue
            line_count += 1
            xmlFile = os.path.join(xmlFileBasePath, line+'.xml')
            object_list = xml_helper.parseXmlFile_countBboxClassNum(
                xmlFile=xmlFile)
            for i_object in object_list:
                label = i_object['name']
                if label in label_count_dict:
                    label_count_dict[label] = label_count_dict[label] + 1
                else:
                    label_count_dict[label] = 1
    if printFlag:
        print("*"*100)
        print("image count in %s is : %d" % (imagelistFile, line_count))
        for key in sorted(label_count_dict.keys()):
            print("%s : %d" % (key, label_count_dict[key]))
    return label_count_dict


"""
    rename image 
"""

def renamePascalImageDataSet(vocpath=None,filePrefix=None):
    if not filePrefix:
        print("filePrefix is None")
        return (False, "filePrefix is None")
    _IMAGE_File_PATH = join(vocpath, 'JPEGImages')
    _XML_FILE_PATH = join(vocpath, 'Annotations')
    if (not os.path.exists(_IMAGE_File_PATH))or(not os.path.exists(_XML_FILE_PATH)):
        infoStr = "%s or %s not exits" % (_IMAGE_File_PATH, _XML_FILE_PATH)
        print(infoStr)
        return (False,infoStr)
    imageCount, imageName_list = utils.getFileCountInDir(
        dirPath=_IMAGE_File_PATH,flag=1)
    xmlCount, xmlName_list = utils.getFileCountInDir(dirPath=_IMAGE_File_PATH,flag=1)
    if imageCount != xmlCount:
        infoStr = "ERROR : image count : %d , xml count : %d"%(imageCount,xmlCount)
        print(infoStr)
        return (False,infoStr)
    newNameIndex = 1
    for i_image in imageName_list:
        oldImageName = i_image
        if not os.path.exists(oldImageName):
            infoStr = "image file : %s not exist" % (oldImageName)
            print(infoStr)
            return (False, infoStr)
        fileParamList = utils.parseFileNameFun(fileName=i_image)
        oldXmlName = os.path.join(_XML_FILE_PATH, fileParamList[1]+'.xml')
        if not os.path.exists(oldXmlName):
            infoStr = "xml file : %s not exist" % (oldXmlName)
            print(infoStr)
            return (False,infoStr)
        newImageName = filePrefix+"-"+str(newNameIndex).rjust(8,'0')+'.jpg'
        newImageAbsoName = os.path.join(_IMAGE_File_PATH, newImageName)
        newXmlName = filePrefix+"-"+str(newNameIndex).rjust(8, '0')+'.xml'
        newXmlAbsoName = os.path.join(_XML_FILE_PATH, newXmlName)
        res,resInfo = renameOneImage(oldImageName=oldImageName, newImageName=newImageAbsoName,
                       oldXmlName=oldXmlName, newXmlName=newXmlAbsoName)
        if not res:
            print(resInfo)
            return (False,"rename %s ERROR"%(i_image))
        else:
            newNameIndex += 1 
    return (True,"rename all image SUCCESS")

def renameOneImage(oldImageName=None,newImageName=None,oldXmlName=None,newXmlName=None):
    # rename image name
    copyImageFileCmdStr = "cp %s %s" % (oldImageName, newImageName)
    res,resInfo = utils.runShellCommandFun(copyImageFileCmdStr)
    if res:
        pass
    else:
        print(resInfo)
        return (False,"copy image : %s error"%(oldImageName))
    # rename xml file name
    copyXmlFileCmdStr = "cp %s %s" % (oldXmlName, newXmlName)
    res,resInfo = utils.runShellCommandFun(copyXmlFileCmdStr)
    if res:
        pass
    else:
        print(resInfo)
        return (False, "copy image : %s error" % (oldImageName))
    # rename xml file : annotation.filename  must rename
    res, resInfo = xml_helper.renameFileNameElement(
        xmlFileName=newXmlName, newFileNameElement=newImageName)
    if not res : # change xml filename element error
        print(resInfo)
        return (False, "change xml fileName element error : %s" % (resInfo))
    else:
        # change xml file filename element success then delete old image and old xml file
        rmOldImageFileCmdStr = "rm %s"%(oldImageName)
        rmOldXmlFileCmdStr = "rm %s"%(oldXmlName)
        res,resInfo = utils.runShellCommandFun(rmOldImageFileCmdStr)
        if not res:
            print(resInfo)
            print("rm old image error ,so must rm just cp new image")
            exit()
        res,resInfo = utils.runShellCommandFun(rmOldXmlFileCmdStr)
        if not res:
            print(resInfo)
            print("rm old xml error ,so must rm just cp new xml")
            exit()
        pass  
    # return new image name
    return (True,"%s ,%s"%(newImageName,newXmlName))


def checkImageReadabilityAndGetMd5(imageName=None,unReadImageLogFileOp=None):
    '''
        check the image readablility ,if can't read or the image is not match the format ,then record it.
        get md5 of the image
    '''
    img = utils.readImage_fun(isUrlFlag=False, imagePath=imageName)
    if np.shape(img) == ():
        unReadImageLogFileOp.write(imageName+'\n')
        unReadImageLogFileOp.flush()
        infoStr = "ReadImageError : %s" % (imageName)
        print(infoStr)
        return (False, infoStr)
    if img.shape[2] != 3:
        unReadImageLogFileOp.write(imageName+'\n')
        unReadImageLogFileOp.flush()
        infoStr = "%s channel is not 3" % (imageName)
        print(infoStr)
        return (False, infoStr)
    res, img_md5 = utils.md5_process(image=img, dataOrNameFlag=0)
    if not res:
        return (False,img_md5)
    else:
        return(True,img_md5)
    pass

# def statisticBboxInfo_one_class(imagelistFile=None, xmlFileBasePath=None, printFlag=True):
#     """
#       imagelistFile is file , per line is a image(xml) file 
#         not include jpg or xml 
#     """
#     line_count = 0
#     label_count_dict = dict()
#     with open(imagelistFile, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             if not line:
#                 continue
#             line_count += 1
#             xmlFile = os.path.join(xmlFileBasePath, line+'.xml')
#             object_list = xml_helper.parseXmlFile_countBboxClassNum(
#                 xmlFile=xmlFile)
#             for i_object in object_list:
#                 label = i_object['name']
#                 if label in ['isis flag', 'islamic flag', 'tibetan flag']:
#                     print(line)
#                 # if label in label_count_dict:
#                 #     label_count_dict[label] = label_count_dict[label] + 1
#                 # else:
#                 #     label_count_dict[label] = 1
#     # if printFlag:
#     #     print("*"*100)
#     #     print("image count in %s is : %d" % (imagelistFile, line_count))
#     #     for key in sorted(label_count_dict.keys()):
#     #         print("%s : %d" % (key, label_count_dict[key]))
#     # return label_count_dict

# def main():
#     # imagelistFile = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/ImageSets/Main/trainval.txt"
#     imagelistFile = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/ImageSets/Main/test.txt"
#     xmlFileBasePath = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/Annotations"
#     # statisticBboxInfo(imagelistFile=imagelistFile,
#     #                       xmlFileBasePath=xmlFileBasePath)
#     statisticBboxInfo_one_class(imagelistFile=imagelistFile,xmlFileBasePath=xmlFileBasePath)
# if __name__ == '__main__':
#     main()
