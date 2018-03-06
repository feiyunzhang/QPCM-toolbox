# --------------------------------------------------------
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xiaoteng Zhang
# --------------------------------------------------------

import _init_paths
import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
import copy
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
cur_path = cur_path[:cur_path.rfind('/')]
update_config(os.path.join(
    cur_path, 'demo/models/config.yaml'))
# add mxnet path
# sys.path.insert(0, os.path.join(
#     cur_path, '/workspace/data/wbb/terror-det-rg-test/mxnet'))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
import random
import json
"""
0,__background__
1,tibetan flag
2,guns
3,knives
5,islamic flag
6,isis flag
"""


def show_boxes(fileOp, image_name, im, dets, classes, scale=1.0):
    color_white = (255, 255, 255)
    # write to terror det rg tsv file
    imageName = image_name
    writeInfo = []
    for cls_idx, cls_name in enumerate(classes):
        if cls_name == "not terror":
            continue
        write_bbox_info = {}
        if cls_name == "tibetan flag":
            write_bbox_info['index'] = 1
        elif cls_name == "guns":
            write_bbox_info['index'] = 2
        elif cls_name == "knives":
            write_bbox_info['index'] = 3
        elif cls_name == "islamic flag":
            write_bbox_info['index'] = 5
        elif cls_name == "isis flag":
            write_bbox_info['index'] = 6
        write_bbox_info['class'] = cls_name

        cls_dets = dets[cls_idx]
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
        for det in cls_dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            one_bbox_write = copy.deepcopy(write_bbox_info)
            bbox_position_list = []
            bbox_position_list.append([bbox[0], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[3]])
            bbox_position_list.append([bbox[0], bbox[3]])
            one_bbox_write["pts"] = bbox_position_list
            one_bbox_write["score"] = float(score)
            writeInfo.append(one_bbox_write)
            cv2.rectangle(im, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color=color, thickness=3)
            cv2.putText(im, '%s %.3f' % (cls_name, score), (bbox[0], bbox[1] + 15),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    fileOp.write("%s\t%s" % (imageName.split('/')[-1], json.dumps(writeInfo)))
    fileOp.write('\n')
    fileOp.flush()
    return im


def parse_args():
    parser = argparse.ArgumentParser(
        description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args


args = parse_args()


def main(tempFileList, fileOp):
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn_dcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    
    out_dir = os.path.join(
        cur_path, 'demo/output/terror-det-rg-data-output/terror-det-v0.9-test/JPEGImages')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # set up class names
    num_classes = 7
    classes = ['tibetan flag', 'guns', 'knives',
               'not terror', 'islamic flag', 'isis flag']

    # load demo data
    image_names = tempFileList
    data = []
    for im_name in image_names:
        im_file = im_name
        print(im_file)
        im = cv2.imread(im_file, cv2.IMREAD_COLOR)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size,
                              stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array(
            [[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names]
            for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max(
        [v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])]
                    for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(
        cur_path + '/demo/models/' + ('rfcn_voc'), 10, process=True)
    #modify by zxt
    #mx.model.save_checkpoint('f1/final', 10, sym, arg_params, aux_params)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[
                                         [(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2]
                  for i in xrange(len(data_batch.data))]
        scores, boxes, data_dict = im_detect(
            predictor, data_batch, data_names, scales, config)

    # test
    # fileOp = open(os.path.join(cur_path, 'terror-det-rg-test-result.txt'), 'w')
    fileOp = fileOp
    for idx, im_name in enumerate(image_names):
        print("begining process %s" % (im_name))
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx, provide_data=[
                                     [(k, v.shape) for k, v in zip(data_names, data[idx])]], provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2]
                  for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(
            predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:,
                              4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        im = cv2.imread(im_name)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_result = show_boxes(fileOp, im_name, im, dets_nms, classes, 1)
        cv2.imwrite(out_dir + im_name.split('/')[-1], im_result)
    print 'done'


def super_main(inputPath= None):
    fileOp = open(os.path.join(cur_path, 'json-result.txt'), 'w')
    data_dir = os.path.join(
        cur_path, 'demo/images/path/to/JPEGImages')
    allFileList = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    beginCount = 0
    endCount = 0
    for i in range(len(allFileList)/500):
        endCount += 500
        tempFileList = allFileList[beginCount:endCount]
        main(tempFileList, fileOp)
        beginCount = endCount
    tempFileList = allFileList[beginCount:]
    main(tempFileList, fileOp)
    pass



if __name__ == '__main__':
    
    super_main()
