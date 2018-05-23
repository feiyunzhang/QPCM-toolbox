# -*- coding:utf-8 -*-
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "../caffe/python")
import caffe
print(caffe.__path__)
from argparse import ArgumentParser
import time
import json
import urllib
from pprint import pprint
"""
    the script used to inference caffe model,classify model
"""

config = {
    "model": {
        "deploy_file": "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/mobilenetv2_deploy_batch.prototxt",
        "weight_file": "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/mobilenetv2-v0.3-t3.caffemodel",
        # csv file ,sep is ,
        "label_file": "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/labels.lst",
        "batch_size": 16
    },
    "dataParam": {
        "resize": 256,
        "input_size": 224,
        "mean_data_list": [103.94, 116.78, 123.68],
        "scala": 0.017
    }
}


only_inference_time = 0


def parser():
    parser = ArgumentParser('caffe model inference')
    parser.add_argument('--inputFileList', dest='inputFileList', help='input file list', required=True,
                        default=None, type=str)
    parser.add_argument('--urlFlag', dest='urlFlag', help='url or local image : urlFlag : 0-local image ,1-url',
                        default=0, type=int)
    parser.add_argument('--gpuId', dest='gpuId', help='The GPU ide to be used', required=True,
                        default=0, type=int)
    return parser.parse_args()


def postProcess(output=None, imagePath=None):
    """
        postprocess net inference result
    """
    output_prob = np.squeeze(output)
    index_list = output_prob.argsort()
    result_dict = dict()
    result_dict['File Name'] = imagePath
    result_dict['Top-1 Index'] = str(index_list[-1])
    result_dict['Top-1 Class'] = str(
        cls_label_dict[int(index_list[-1])].split(' ')[0])
    result_dict['Confidence'] = str(output_prob[index_list[-1]])
    resultFileOp.write(json.dumps(result_dict))
    resultFileOp.write('\n')
    resultFileOp.flush()


def readImage_fun(isUrlFlag=False, imagePath=None):
    im = None
    if isUrlFlag:
        try:
            data = urllib.urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            # print("Read Url Exception : %s" % (imagePath))
            im = None
        else:
            im = cv2.imdecode(nparr, 1)
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if np.shape(im) == ():
        # print("waringing info : %s can't be read" % (imagePath))
        return None
    return im


def init_models():
    caffe.set_mode_gpu()
    caffe.set_device(args.gpuId)
    deployName = config['model']['deploy_file']
    modelName = config['model']['weight_file']
    net_cls = caffe.Net(deployName, modelName, caffe.TEST)
    return net_cls


def getImageList():
    global resultFileOp
    image_list = []
    with open(args.inputFileList, 'r') as f:
        image_list = [i.strip() for i in f.readlines() if i.strip()]
    #create result file op
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    resultFile = args.inputFileList+'-'+time_str+'-result.json'
    resultFileOp = open(resultFile, 'w')
    return image_list


def preProcess(oriImage=None):
    img = cv2.resize(
        oriImage, (config['dataParam']['resize'], config['dataParam']['resize']))
    # img = img.astype(np.float32, copy=True)
    img -= np.array([[config['dataParam']['mean_data_list']]])
    img = img * config['dataParam']['scala']
    short_edge = min(img.shape[:2])
    crop_size = config['dataParam']['input_size']
    if short_edge < crop_size:
        return None
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    img = img[yy: yy + crop_size, xx: xx + crop_size]
    img = img.transpose((2, 0, 1))
    return img


def infereneAllImage(net_cls=None, imageList=None, urlFlag=None):
    global only_inference_time
    t_begin = time.time()
    batch_image_data = []
    batch_image_path = []
    batch_count = 0
    for image_path in imageList:
        oriImg = readImage_fun(isUrlFlag=urlFlag, imagePath=image_path)
        if np.shape(oriImg) == ():
            print("ReadImageError : %s" % (image_path))
            continue
        if oriImg.shape[2] != 3:
            print("%s channel is not 3" % (image_path))
            continue
        img = preProcess(oriImage=oriImg)
        batch_image_data.append(img)
        batch_image_path.append(image_path)
        batch_count += 1
        if batch_count == config['model']['batch_size']:
            for index, i_data in enumerate(batch_image_data):
                net_cls.blobs['data'].data[index] = i_data
            t_1 = time.time()
            output = net_cls.forward()
            t_2 = time.time()
            only_inference_time += (t_2 - t_1)
            for i, i_output in enumerate(output['prob']):
                postProcess(output=i_output, imagePath=batch_image_path[i])
            batch_image_data = []
            batch_image_path = []
            batch_count = 0
    # process not enough 20
    last_batch_size = len(batch_image_data)
    if last_batch_size == 0:
        pass
    else:
        for index, i_data in enumerate(batch_image_data):
            net_cls.blobs['data'].data[index] = i_data
        t_1 = time.time()
        output = net_cls.forward()
        t_2 = time.time()
        only_inference_time += (t_2 - t_1)
        for i in range(len(batch_image_data)):
            i_output = output['prob'][i]
            postProcess(output=i_output, imagePath=batch_image_path[i])
    t_end = time.time()
    print("batch size is : %d" % (config['model']['batch_size']))
    print("one image : only inference time is : %f" %
          (only_inference_time/len(imageList)))
    print("one image : process time : %f" % ((t_end-t_begin)/len(imageList)))


args = parser()
cls_label_dict = dict()
resultFileOp = None


def get_label_list():
    global cls_label_dict
    label_file = config['model']['label_file']
    if not label_file:
        print("label file is null")
        exit()
    with open(label_file, 'r') as f:
        for line in f.readlines():
            if line.strip():
                index, name = line.split(',')
                cls_label_dict[int(index)] = name
            else:
                continue
    pass


def temp_init():
    get_label_list()
    print('*'*20+"label file"+'*'*20)
    pprint(cls_label_dict)
    pass


def main():
    temp_init()  # serve different need
    urlFlag = True if args.urlFlag == 1 else False
    net_cls = init_models()
    imageList = getImageList()
    infereneAllImage(net_cls=net_cls, imageList=imageList, urlFlag=urlFlag)


if __name__ == '__main__':
    main()


"""
python terror-class-inference-demo.py \
--inputFileList /workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/test-data/local_images.list \
--urlFlag 0 \
--gpuId 0 \
"""
