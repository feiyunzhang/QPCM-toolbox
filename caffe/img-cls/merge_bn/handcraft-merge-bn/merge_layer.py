import numpy as np
import sys
import os
caffe_root = '/workspace/data/BK/refineDet-Dir/RefineDet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

train_proto = './models/deploy-final.prototxt'
# should be your snapshot caffemodel
train_model = '/workspace/data/BK/cls-mix-det-Dir/models/mix_res18_det_and_cls_0606/lib/models/Final_model.caffemodel'

deploy_proto = './models/deploy-final-mergebn.prototxt'
save_model = '/workspace/data/BK/cls-mix-det-Dir/models/mix_res18_det_and_cls_0606/lib/models/Final_model-mergebn-1.caffemodel'

import merge_layer_config as layer_config
MERGE_LAYER_DICT_CONFIG = layer_config.merge_conv_layer_dict


def getAllBnScaleLayers(layer_dict_config=None):
    bn_scale_layers = []
    for key, value in layer_dict_config.iteritems():
        for i_value in value:
            bn_scale_layers.append(i_value)
    return bn_scale_layers


def merge_net(net, nob):
    '''
        net is normal net
        nob is merged net
    '''
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    bn_scale_layers = getAllBnScaleLayers(layer_dict_config=MERGE_LAYER_DICT_CONFIG)
    for i_layer_name in net.params.iterkeys():
        i_layer_data = net.params[i_layer_name]
        if not isinstance(i_layer_data, caffe._caffe.BlobVec):  # check layer type 
            print("layer : {layer} is not caffe._caffe.BlobVec".format(layer=i_layer_name))
            break   
        if i_layer_name in bn_scale_layers: # bn or scale ,not save to merged net
            print("layer : {layer} is bn or scale , not process".format(layer=i_layer_name))
            continue   
        print("layer name is : {layer}".format(layer=i_layer_name))
        if i_layer_name not in MERGE_LAYER_DICT_CONFIG.keys():  
            # the layer not contains bn ,scale ,so just save the layer to merge net
            print("{layer} don't need merge bn and scale to conv".format(
                layer=i_layer_name))
            for i, w in enumerate(i_layer_data):
                nob.params[i_layer_name][i].data[...] = w.data
        else: 
             # just merge layer in config dict
            bn_layer_name = MERGE_LAYER_DICT_CONFIG[i_layer_name][0]
            scale_layer_name = MERGE_LAYER_DICT_CONFIG[i_layer_name][1]
            bn_layer_data = net.params[bn_layer_name]
            scale_layer_data = net.params[scale_layer_name]
            conv_w_old = i_layer_data[0].data
            conv_w_old_channels = conv_w_old.shape[0]
            conv_bias_old = np.zeros(conv_w_old_channels)
            if len(i_layer_data) > 1:
                conv_bias_old = i_layer_data[1].data
            mean = bn_layer_data[0].data # bn - mean
            var = bn_layer_data[1].data  # bn - var
            scalef = bn_layer_data[2].data 

            scales = scale_layer_data[0].data
            shift = scale_layer_data[1].data
            if scalef != 0:
                scalef = 1. / scalef
            mean = mean * scalef
            var = var * scalef
            rstd = 1. / np.sqrt(var + 1e-5)
            rstd1 = rstd.reshape((conv_w_old_channels, 1, 1, 1))
            scales1 = scales.reshape((conv_w_old_channels, 1, 1, 1))
            conv_w_new = conv_w_old * rstd1 * scales1
            conv_bias_new = (conv_bias_old - mean) * rstd * scales + shift
            nob.params[i_layer_name][0].data[...] = conv_w_new
            if len(nob.params[i_layer_name]) > 1: # error , the merged net file , conv should contains bias 
                nob.params[i_layer_name][1].data[...] = conv_bias_new


caffe.set_mode_gpu()
caffe.set_device(7)
net = caffe.Net(train_proto, train_model, caffe.TRAIN)
net_deploy = caffe.Net(deploy_proto, caffe.TEST)

merge_net(net, net_deploy)
net_deploy.save(save_model)
