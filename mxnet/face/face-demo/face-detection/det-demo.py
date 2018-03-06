import numpy as np  
import sys,os  
import cv2
caffe_root = 'path/to/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time

def preprocess(src):
    img = cv2.resize(src, (512, 512))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(data_dir, imgfile, out_dir):
    origimg = cv2.imread(os.path.join(data_dir, imgfile))
    origin_h,origin_w,ch = origimg.shape
    starttime = time.time()
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = img
    out = net.forward()  
    endtime = time.time()
    print 'speed: {:.3f}s / iter'.format(endtime - starttime)

    box, conf, cls = postprocess(origimg, out)
    numbox = len(box)
    if numbox == 0:
        print 'no face detect'
    else:
        for i in range(numbox):
            #crop = origimg[int(box[0][1]):int(box[0][3]), int(box[0][0]):int(box[0][2])]
            crop = origimg[max(0, int(box[i][1])-5):min(int(box[i][3])+5, origin_h), max(0, int(box[i][0])-5):min(int(box[i][2])+5, origin_w)]
            if int(crop.shape[0])*int(crop.shape[1]) < 2000:
                print 'face pixel is not clear'
                continue
            else:
                cv2.imwrite(os.path.join(out_dir, imgfile), crop)
                break




if __name__ == '__main__':
	net_file= 'det-model/deploy.prototxt'  
	caffe_model='det-model/model.caffemodel'  


	if not os.path.exists(caffe_model):
		print("deploy.affemodel does not exist,")
		print("use merge_bn.py to generate it.")
		exit()
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(net_file,caffe_model,caffe.TEST) 

 
	data_dir = 'test_images/'
        out_dir = 'out_images/'


	imgpath = os.listdir(data_dir)
	for imgfile in imgpath:
		detect(data_dir, imgfile, out_dir)

