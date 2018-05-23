import os
import sys
import json
import numpy as np
import hashlib
import time
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import cv2

def runShellCommandFun(cmdStr=None):
    if not cmdStr:
        return (False,"the commad is NULL")
    res = os.system(cmdStr)
    if res == 0: # cmd run success
        return (True,"%s run success"%(cmdStr))
    else:
        return (False,"%s run error , result code : %s"%(cmdStr,str(res)))


def getTimeFlag(flag=0):
    if flag==0:
        formatStr = "%Y-%m-%d-%H"
    elif flag==1:
        formatStr = "%Y%m%d"
    return time.strftime(formatStr, time.localtime())



def getFileCountInDir(dirPath=None,flag=0):
    if flag == 0: # return file list : element just file name
        file_list = [i for i in os.listdir(dirPath) if i[0] != '.' and os.path.isfile(os.path.join(dirPath, i))]
        return [len(file_list), sorted(file_list)]
    elif flag == 1: # return file list : element is absolute file path
        file_list = [os.path.join(dirPath, i) for i in os.listdir(
            dirPath) if i[0] != '.' and os.path.isfile(os.path.join(dirPath, i))]
        return [len(file_list), sorted(file_list)]
    pass


def parseFileNameFun(fileName=None):
    """
        return :
            filePath : just folder base path
            justFileName : just file name not include folder path
            absoluteFilePathNotIncludePosfix: absolute file path and file name but not include posfix
    """
    justFileName = os.path.split(fileName)[-1]
    filePath = os.path.split(fileName)[0]
    if '.' in justFileName:
        justFileName = justFileName[:justFileName.rfind('.')]
    return [filePath, justFileName, os.path.join(filePath, justFileName)]
    pass

def readImage_fun(isUrlFlag=None, imagePath=None):
    """
        isUrlFlag == True , then read image from url
        isUrlFlag == False , then read image from local path
    """
    im = None
    if isUrlFlag == True:
        try:
            data = urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            im = None
        else:
            im = cv2.imdecode(nparr, 1)
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if np.shape(im) == ():
        return None
    return im

def md5_process(image=None,dataOrNameFlag=None):
    """
        read image , get md5 
        dataOrNameFlag == 0: image is imageData
        dataOrNameFlag  == 1 : image is loca image file name
        dataOrNameFlag == 2 : image  is url 
    """
    if dataOrNameFlag == 0:
        imageData = image
    elif dataOrNameFlag == 1:
        imageData = readImage_fun(isUrlFlag=False, imagePath=image)
    elif dataOrNameFlag == 2:
        imageData = readImage_fun(isUrlFlag=True, imagePath=image) 
        pass
    if imageData == None:
        return (False, "imageData is None")
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: imageData.read(4096), b""):
        hash_md5.update(chunk)
    return (True,hash_md5.hexdigest())



    
