#coding=utf-8
import numpy as np
import cv2
import sys
sys.path.insert(0, "models/caffe/python")
import caffe
import argparse
import json
from cfg import Config
import urllib

def init_models():
    if Config.PLATFORM == "GPU":
        caffe.set_mode_gpu()
        caffe.set_device(Config.TEST_GPU_ID)
    else:
        caffe.set_mode_cpu()
    # initialize the cls model
    cls_mod = caffe.Net(Config.CLS_NET_DEF_FILE,Config.CLS_MODEL_PATH,caffe.TEST)
    return cls_mod

def txt_to_dict():
    label_index = np.loadtxt(Config.CLS_LABEL_INDEX,str,delimiter='\n')
    cls_dict = {}
    for index in label_index:
        cls_dict.update({index.split(' ')[0]:index.split(' ')[1]})
    return cls_dict

def classify_flow(url,label_cls):
    flow = {
        "url": url,
        "type": "image",
        "label": {"class": {"terror":label_cls}}
    }
    return flow

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



def save_json_in_text(json_dict=None, text_path=None):
    with open(text_path,'w') as f:
        json_result = json.dumps(json_dict, ensure_ascii=False)
        print json_result
        f.write(json_result)
        f.flush()
        f.close()
    pass


def process_image_fun(net_cls=None, image_path=None,cls_dict=None):
    origimg = cv2.imread(image_path)
    if np.shape(origimg) != ():
        label_cls, score_cls = cls_process(net_cls, origimg)
        json_dict={}
        cla_index = str(label_cls[0])
        if float(score_cls[0]) > Config.CLS_CONFIDENCE_THRESH:
            json_dict['img']=image_path
            json_dict['img_type']=cls_dict.get(cla_index)
        else:
            json_dict['img']=image_path
            json_dict['img_type']="others"
        save_json_in_text(json_dict=json_dict,text_path=image_path+'_cls.json')

def process_image_urllist(net_cls=None,urllist_filePath=None,cls_dict=None):
    saveJsonFile = urllist_filePath+'_cls_result.json'
    with open(urllist_filePath,'r') as read_file,open(saveJsonFile,'w') as write_file:
        for url_line in read_file.readlines():
            if len(url_line.strip()) == 0:
                continue
            data = urllib.urlopen(url_line.strip()).read()
            nparr = np.fromstring(data,np.uint8)
            origimg = cv2.imdecode(nparr,1)
            if np.shape(origimg) != ():
                label_cls, score_cls = cls_process(net_cls, origimg)
                json_dict={}
                
                cla_index = str(label_cls[0])
                json_dict = classify_flow(url_line.strip(),cls_dict.get(cla_index) if float(score_cls[0]) > Config.CLS_CONFIDENCE_THRESH else "others")
                write_file.write(json.dumps(json_dict, ensure_ascii=False))
                write_file.write('\n')
                write_file.flush()
                pass
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='AtLab Label Image!',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image', help='input image', default=None, type=str)
    parser.add_argument('--imagelist', help='input image list', default=None, type=str)
    parser.add_argument('--urllist', help='input image url list', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cls_mod = init_models()
    cls_dict = txt_to_dict()
    if args.image is not None:
        image_path = args.image
        process_image_fun(net_cls=cls_mod,image_path=image_path,cls_dict=cls_dict)
    elif args.imagelist is not None:
        with open(args.imagelist,'r') as f:
            image_paths = f.readlines()
        for image_path in image_paths:
            image_path = image_path.strip()
            process_image_fun(net_cls=cls_mod,image_path=image_path,cls_dict=cls_dict)
    elif args.urllist is not None:
        process_image_urllist(net_cls=cls_mod,urllist_filePath=args.urllist,cls_dict=cls_dict)
        pass
