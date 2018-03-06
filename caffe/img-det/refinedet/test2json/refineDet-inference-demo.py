'''
In this example, we will load a RefineDet model and use it to detect objects.
'''
import numpy as np
import sys
import os
import cv2
caffe_root = 'path/to/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import time
import argparse
import random
import skimage.io as io

from google.protobuf import text_format
from caffe.proto import caffe_pb2
import json


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(
                    [labelmap.item[i].display_name, labelmap.item[i].label])
                break
        assert found == True
    return labelnames


def ShowResults(img=None,image_file=None, outputDir=None, results=None, labelmap=None, rgFileOp=None, threshold=0.6, save_fig=False):
    # plt.imshow(img)
    # plt.axis('off')
    # ax = plt.gca()
    # results = [bbox,conf,cls]
    img = img  # image_file absolute path
    color_white = (255, 255, 255)
    num_classes = len(labelmap.item) - 1
    # colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    rgWriteInfo = []
    all_bbox = results[0]
    conf = results[1]
    cls = results[2]
    for i in range(0, all_bbox.shape[0]):
        bbox_dict = dict()
        score = conf[i]
        if threshold and score < threshold:
            continue
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
        label = int(cls[i])
	bbox = all_bbox[i]
        name_label = get_labelname(labelmap, label)[0]
        name = name_label[0]
        xmin = int(round(bbox[0]))
        ymin = int(round(bbox[1]))
        xmax = int(round(bbox[2]))
        ymax = int(round(bbox[3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      color=color_white, thickness=3)
        cv2.putText(img, "%s-%.3f" % (name, score), (xmin, ymin + 25),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)

        bbox_position_list = []
        bbox_position_list.append([xmin, ymin])
        bbox_position_list.append([xmax, ymin])
        bbox_position_list.append([xmax, ymax])
        bbox_position_list.append([xmin, ymax])
        bbox_dict['pts'] = bbox_position_list
        bbox_dict['score'] = float(score)
        bbox_dict['index'] = name_label[1]
        bbox_dict['class'] = name
        rgWriteInfo.append(bbox_dict)
    if save_fig:
        outputName = os.path.join(
            args.testDataOutputBasePath, image_file.split('/')[-1])
        cv2.imwrite(outputName, img)
    # add regression info
    rgFileOp.write("%s\t%s\n" %
                   (image_file.split('/')[-1], json.dumps(rgWriteInfo)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='test demo for refineDet Model')
    parser.add_argument('--modelBasePath',
                        dest='modelBasePath', default=None, type=str)
    parser.add_argument('--caffemodelName',
                        dest='caffemodelName', default=None, type=str)
    parser.add_argument('--prototxtName',
                        dest='prototxtName', default=None, type=str)
    parser.add_argument('--labelmapFile',
                        dest='labelmapFile', default=None, type=str)
    parser.add_argument('--testDataInputBasePath',
                        dest='testDataInputBasePath', default=None, type=str)
    parser.add_argument('--testImageListFile',
                        dest='testImageListFile', default=None, type=str)
    parser.add_argument('--testImageListFile_ImageBasePath',
                        dest='testImageListFile_ImageBasePath', default=None, type=str)

    # if saveImageFlag == True ; then must write testDataOutputBasePath
    parser.add_argument('--saveImageFlag',
                        dest='saveImageFlag', default=False, type=bool)
    parser.add_argument('--testDataOutputBasePath',
                        dest='testDataOutputBasePath', default=None, type=str)
    # rgTestResultFile is absolute file
    parser.add_argument('--rgTestResultFile',
                        dest='rgTestResultFile', default=None, type=str)
    parser.add_argument('--gpuId', dest='gpuId', default=0, type=int)

    return parser.parse_args()


def preprocess(img=None,img_resize=None):
    img = cv2.resize(img, (img_resize, img_resize))
    img = img.astype(np.float32,copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    return img


def postprocess(img=None, out=None):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)




args = parse_args()
if __name__ == '__main__':
    # gpu preparation
    gpu_id = int(args.gpuId)
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # load labelmap
    labelmap_file = os.path.join(args.modelBasePath, args.labelmapFile)
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # load model
    model_def = os.path.join(args.modelBasePath, args.prototxtName)
    model_weights = os.path.join(args.modelBasePath, args.caffemodelName)
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_weights or '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    # regression file
    rgFileOp = open(args.rgTestResultFile, 'w')

    testImageAbsolutePath = []

    if args.testDataInputBasePath != None:
        for im_name in os.listdir(args.testDataInputBasePath):
            image_file = os.path.join(args.testDataInputBasePath, im_name)
            testImageAbsolutePath.append(image_file)
    elif args.testImageListFile != None:
        with open(args.testImageListFile, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    line = line.strip()+'.jpg'
                    image_file = os.path.join(
                        args.testImageListFile_ImageBasePath, line)
                    testImageAbsolutePath.append(image_file)
    # im_names = os.listdir('examples/images')
    for im_name in testImageAbsolutePath:
        print("process : %s" % (im_name))
        origimg = cv2.imread(im_name)
        origin_h, origin_w, ch = origimg.shape
        img = preprocess(img=origimg, img_resize=img_resize)
        print(img.shape)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        net.blobs['data'].data[...] = img
        starttime = time.time()
        out = net.forward()
        box, conf, cls = postprocess(img=origimg, out=out)
        num = len(box)
        endtime = time.time()
        per_time = float(endtime - starttime)
        print('speed: {:.3f}s / iter'.format(endtime - starttime))
        # show result
        # ShowResults(image, image_file, outputDir,result, labelmap, 0.6, save_fig=False)
        if args.saveImageFlag == True:
            if not os.path.exists(args.testDataOutputBasePath):
                os.makedirs(args.testDataOutputBasePath)
        # image_file is absolute path
        ShowResults(img=origimg, image_file=im_name, outputDir=args.testDataOutputBasePath,
                    results=[box, conf, cls], labelmap=labelmap, rgFileOp=rgFileOp, threshold=0.6, save_fig=args.saveImageFlag)


"""
# regression
nohup \
python -u /workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/train-terror-0.9/inference-Dir/refineDet-inference-demo-rg-writeFile.py \
--modelBasePath /workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/train-terror-0.9/models \
--caffemodelName  terror_refinedet_resnet102_320x320_terror-v0.9_iter_80000.caffemodel \
--prototxtName  deploy.prototxt \
--labelmapFile labelmap_voc.prototxt  \
--testImageListFile /workspace/data/BK/terror-model-test-Dir/data/terror-det-v0.9-test/TERROR-DETECT-V0.9/ImageSets/Main/test.txt \
--testImageListFile_ImageBasePath /workspace/data/BK/terror-model-test-Dir/data/terror-det-v0.9-test/TERROR-DETECT-V0.9/JPEGImages \
--rgTestResultFile /workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/train-terror-0.9/inference-Dir/regressionResult.tsv \
--gpuId  0 \
> /workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/train-terror-0.9/inference-Dir/log.log \
2>&1 &
"""

