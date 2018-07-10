import numpy as np  
import sys,os  
caffe_root = 'caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

det_proto = 'prototxt/C-det.prototxt'  
det_model = 'models/C-stage-terror-face_iter_60000.caffemodel'

terror_proto = 'prototxt/D-terror.prototxt'  
terror_model = 'models/D-stage-terror-cls_iter_45000.caffemodel'

pulp_proto = 'prototxt/E-pulp.prototxt'
pulp_model = 'models/E-stage-pulp-cls_iter_140000.caffemodel'

final_proto = 'prototxt/final.prototxt'  
final_model = 'Final_model.caffemodel'

def merge_det(net, nob, key):
    '''
    merge det to final model
    '''
    if key in net.params.iterkeys():
        conv = net.params[key]
        print 'det-layer:', key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data

def merge_terror(net, nob, key):
    '''
    merge terror to final model
    '''
    if key in net.params.iterkeys():
        conv = net.params[key]
        print 'terror-layer:', key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data

def merge_pulp(net, nob, key):
    '''
    merge pulp to final model
    '''
    if (key.split('_pulp')[0] + '_cls') in net.params.iterkeys():
        new_key = key.split('_pulp')[0] + '_cls'
        conv = net.params[new_key]
        print 'pulp_layer:', key, new_key
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data
    elif 'fc7-pulp' in net.params.iterkeys():
        conv = net.params['fc7-pulp']
        print 'pulp_extra_layer:', 'fc7-pulp'
        for i, w in enumerate(conv):
            nob.params[key][i].data[...] = w.data


if __name__ == '__main__':
    net_det = caffe.Net(det_proto, det_model, caffe.TRAIN)  
    net_terror = caffe.Net(terror_proto, terror_model, caffe.TRAIN) 
    net_pulp = caffe.Net(pulp_proto, pulp_model, caffe.TRAIN)

    net_final = caffe.Net(final_proto, caffe.TEST)  
    for final_key in net_final.params.iterkeys():
        if final_key in net_det.params.iterkeys():
            merge_det(net_det, net_final, final_key)
        elif final_key in net_terror.params.iterkeys():
            merge_terror(net_terror, net_final, final_key)
        else:
            merge_pulp(net_pulp, net_final, final_key)
    net_final.save(final_model)
