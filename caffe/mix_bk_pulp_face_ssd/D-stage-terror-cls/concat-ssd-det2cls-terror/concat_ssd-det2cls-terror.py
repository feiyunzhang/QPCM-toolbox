import numpy as np  
import sys,os  
caffe_root = 'caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

det_proto = 'deploy-det-ssd.prototxt'  
det_model = 'models/C-stage-terror-face_iter_60000.caffemodel'

cls_proto = 'deploy-cls-bk.prototxt'  
cls_model = 'models/A-stage-terror-cls_iter_20000.caffemodel'

final_proto = 'deploy-cls-bk-D.prototxt'  
final_model = 'D-concat-terror-cls-pretrain.caffemodel'

def merge_det(net, nob, key):
    '''
    merge det to final model
    '''
    if key in net.params.iterkeys():
        conv = net.params[key]
        print 'det:', key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data

def merge_cls(net, nob, key):
    '''
    merge cls to final model
    '''
    if key in net.params.iterkeys():
        conv = net.params[key]
        print 'cls:', key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data
    elif key.split('_cls')[0] in net.params.iterkeys():
        new_key = key.split('_cls')[0]
        conv = net.params[new_key]
        print 'new_key:', new_key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data


if __name__ == '__main__':
    net_det = caffe.Net(det_proto, det_model, caffe.TRAIN)  
    net_cls = caffe.Net(cls_proto, cls_model, caffe.TRAIN) 
    
    net_final = caffe.Net(final_proto, caffe.TEST)  
    for final_key in net_final.params.iterkeys():
        if final_key in net_det.params.iterkeys():
            merge_det(net_det, net_final, final_key)
        else:
            merge_cls(net_cls, net_final, final_key)
    net_final.save(final_model)
