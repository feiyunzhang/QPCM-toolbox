from ctypes import *
import math
import random
import numpy as np
import os
import cv2
import time

lib = CDLL("lib/libdarknet_gpu.so", RTLD_GLOBAL)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

    

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def plot_image(plot_im, num, coco_names, boxes, probs):
    color_white = (255, 255, 255)
    for i in range(1):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        for j in range(num):
            if probs[j][i] > 0:
                cv2.rectangle(plot_im, (int(boxes[j].x-boxes[j].w/2.0), int(boxes[j].y-boxes[j].h/2.0)), (int(boxes[j].x+boxes[j].w/2.0), int(boxes[j].y+boxes[j].h/2.0)), color=color, thickness=3)
                cv2.putText(plot_im, '%s %.3f' % (coco_names[i], probs[j][i]), (int(boxes[j].x-boxes[j].w/2.0), int(boxes[j].y-boxes[j].h/2.0) + 15), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return plot_im

def detect(net, coco_names, image, thresh=.4, hier_thresh=.4, nms=.3):
    im = load_image(image, 0, 0)
    starttime = time.time()
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    endtime = time.time()
    print 'speed: {:.3f}s / iter'.format(endtime - starttime)

    plot_im = cv2.imread(image)
    new_plot_im = plot_image(plot_im, num, coco_names, boxes, probs)
    return new_plot_im
    
if __name__ == "__main__":
    net = load_net("models/yolo.cfg", "models/yolo.weights", 0)
    coco_names = np.loadtxt('lib/coco_names.txt', str, delimiter='\n')

    data_dir = 'images/'
    out_dir = 'output/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgpath = os.listdir(data_dir)
    for imgfile in imgpath:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(imgfile)
        img = os.path.join(data_dir, imgfile)
        plot_im = detect(net, coco_names, img)
        cv2.imwrite(os.path.join(out_dir,imgfile), plot_im)
    

