#coding=utf-8  
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "lib/caffe/python")
import caffe
from argparse import ArgumentParser
import time

def parser():
    parser = ArgumentParser('AtLab Label Image!')
    parser.add_argument('--images',dest='im_path',help='Path to the image',
                        default='test/',type=str)
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

def cls_process(net_cls, img):
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32, copy=True)
    img -= np.array([[[103.94,116.78,123.68]]])
    img = img * 0.017

    img = center_crop(img, 225)

    img = img.transpose((2, 0, 1))
    net_cls.blobs['data'].data[...] = img
    out = net_cls.forward() 
    score = out['prob'][0] 
    sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
    label_cls = [sort_pre[-j][0] for j in range(1,2)]
    score_cls = [sort_pre[-j][1] for j in range(1,2)]
    return label_cls, score_cls


if __name__ == '__main__':
    args = parser()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net_cls = caffe.Net("lib/models/cls_test.prototxt", "lib/models/t7_32000.caffemodel", caffe.TEST)

    cls_list = np.loadtxt('test.txt', str, delimiter='\n')      
    filename = os.listdir(args.im_path)
    filename.sort()
    result = open('test_result_1.7w.txt','w')
    score_top1 = 0
    index = 1
    for i in range(len(filename)):
        origimg = cv2.imread(os.path.join(args.im_path, filename[i]))
        if np.shape(origimg) != ():
            starttime = time.time()

            label_cls, score_cls = cls_process(net_cls, origimg)
            endtime = time.time()
            print 'speed: {:.3f}s / iter'.format(endtime - starttime)
            if int(cls_list[i].split(' ')[1]) == label_cls[0]:
                score_top1 = score_top1 + 1
            index  = index + 1
            print 'Top1-acc:', score_top1/float(index)
            result.write(filename[i] + ' ' + str(label_cls[0]) + '\n')
    result.close()
    Top1_acc = float(score_top1)/2498.0
    print 'Top1-acc', Top1_acc


         





    
