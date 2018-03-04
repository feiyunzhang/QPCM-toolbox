import numpy as np  
import sys,os  
import cv2
caffe_root = '../demo/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time
from fast_rcnn.test import multi_scale_test_net_512


if __name__ == '__main__':
    net_file= '../demo/models/single_test_deploy.prototxt'  
    caffe_model='../demo/models/coco_refinedet_resnet101_512x512_final.caffemodel'  
    if not os.path.exists(caffe_model):
        print("deploy.affemodel does not exist,")
        print("use merge_bn.py to generate it.")
        exit()
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(net_file,caffe_model,caffe.TEST) 
    coco_names = np.loadtxt('../lib/coco_names.txt', str, delimiter='\n')
 
    data_dir = '../demo/images/'
    out_dir = '../demo/output/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgpath = os.listdir(data_dir)
    for imgfile in imgpath:
        result = open(out_dir + '/' + imgfile.split('.')[0] + '.txt', 'w+') 
        starttime = time.time()
        cls_det = multi_scale_test_net_512(net, data_dir + '/' + imgfile, soft_nms=False)
        endtime = time.time()
        print 'speed: {:.3f}s / iter'.format(endtime - starttime)
        result.write(imgfile.split('.')[0] + '\n')
        for j in range(1,81):
            if len(cls_det[j][0]) != 0:
                for i in range(len(cls_det[j][0])):
    	            result.write(str(coco_names[j-1]) + ' ' + str(int(cls_det[j][0][i][0])) + ' ' + str(int(cls_det[j][0][i][1])) + ' ' + str(int(cls_det[j][0][i][2])) + ' ' + str(int(cls_det[j][0][i][3])) + ' ' + str(cls_det[j][0][i][4]) + '\n' )
        result.close()

