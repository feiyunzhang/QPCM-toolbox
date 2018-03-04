#coding=utf-8  
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "lib/caffe/python")
import caffe
from argparse import ArgumentParser
import time
import json

def parser():
    parser = ArgumentParser('AtLab Label Image!')
    parser.add_argument('--img_file',dest='img_file',help='Path to the image',
                        default='test.lst',type=str)
    parser.add_argument('--root',dest='data_root',help='Path to the image',
                        default='val/',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    return parser.parse_args()

def center_crop(img, crop_size): 
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]

def single_img_process(net_cls, im_path, ori_img, label_list):
    img = cv2.imread(os.path.join(im_path, ori_img))
    if np.shape(img) != ():
        starttime = time.time()
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32, copy=True)
        img -= np.array([[[103.94,116.78,123.68]]])
        img = img * 0.017

        img = center_crop(img, 225)

        img = img.transpose((2, 0, 1))
        net_cls.blobs['data'].data[...] = img
        output = net_cls.forward()
        lst_result = list()
        output_prob = np.squeeze(output['prob'][0])

        index_list = output_prob.argsort()
        rate_list = output_prob[index_list]

        result_dict = dict()
        result_dict['File Name'] = ori_img
        result_dict['Top-1 Index'] = index_list[-1]

        result_dict['Top-1 Class'] = label_list[int(index_list[-1])].split(' ')[1]

        result_dict['Confidence'] = [str(i) for i in list(output_prob)]
        lst_result.append(result_dict)
    return lst_result


if __name__ == '__main__':
    args = parser()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net_cls = caffe.Net("lib/models/deploy.prototxt", "lib/models/se-res50-hiv-v0.3-t1_iter_78000.caffemodel", caffe.TEST)

    cls_list = np.loadtxt('lib/labels.lst', str, delimiter='\n')
    test_list = np.loadtxt(args.img_file, str, delimiter='\n')         

    dict_result = {}

    for i in range(len(test_list)):
        starttime = time.time()
        dict_result_tmp = single_img_process(net_cls, args.data_root, test_list[i], cls_list)
        endtime = time.time()
        print 'speed: {:.3f}s / iter'.format(endtime - starttime)
        for item in dict_result_tmp:
            dict_result[os.path.basename(item['File Name'])] = item
    log_result = open('log.json', 'w')
    json.dump(dict_result, log_result, indent=4)
    log_result.close()
