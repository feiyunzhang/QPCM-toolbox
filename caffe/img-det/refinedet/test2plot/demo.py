# ------------------------------------------
# AtLab_SSD_Mobilenet_V0.1
# Demo
# by Zhang Xiaoteng
# ------------------------------------------
import numpy as np  
import sys
import os  
from argparse import ArgumentParser
if not 'demo/caffe/python' in sys.path:
    sys.path.insert(0,'demo/caffe/python') 
import caffe  
import time
import cv2
import random
def parser():
    parser = ArgumentParser('AtLab RD-Terror Demo!')
    parser.add_argument('--images',dest='im_path',help='Path to the image',
                        default='demo/images',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSD caffe test prototxt',
                        default='demo/models/deploy-old.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSD trained caffemodel',
                        default='demo/models/rd-bk-mobilenetv2-c1_iter_60000.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='demo/output/',type=str)
    return parser.parse_args()

def preprocess(src):
    img = cv2.resize(src, (320,320))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.94, 116.78, 123.68]]])
    img = img * 0.017
    return img


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def plot_image(plot_im, num, coco_names, box, conf, cls):
    color_white = (0, 0, 0)
    for i in range(5):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        for j in range(num):
            if int(cls[j]) == (i+79):
                box[j] = map(int, box[j])
                cv2.rectangle(plot_im, (box[j][0], box[j][1]), (box[j][2], box[j][3]), color=color, thickness=3)
                cv2.putText(plot_im, '%s %.3f' % (coco_names[i], float(conf[j])), (box[j][0], box[j][1] + 15), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return plot_im

def detect(net, im_path, coco_names):
    origimg = cv2.imread(im_path)
    origin_h,origin_w,ch = origimg.shape
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    starttime = time.time()
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    num = len(box)
    endtime = time.time()
    per_time = float(endtime - starttime)
    print 'speed: {:.3f}s / iter'.format(endtime - starttime)

    new_plot_im = plot_image(origimg, num, coco_names, box, conf, cls)
                      
    return new_plot_im, per_time
       
  

if __name__ == "__main__":
    args = parser()
    coco_names = np.loadtxt('demo/models/bk_names.txt', str, delimiter='\n')
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'AtLab-RD-Terror'
    print('Done!')
    
    totle_time = 0
    for image in os.listdir(args.im_path):
        img = os.path.join(args.im_path,image)
        plot_im, per_time = detect(net,img,coco_names)
        totle_time = totle_time + per_time
        cv2.imwrite(os.path.join(args.out_path,image), plot_im)
    print totle_time

