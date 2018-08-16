import sys
import argparse
import find_mxnet
import find_caffe
import mxnet as mx
import caffe
import pdb
import json
import numpy as np
import copy


def merge_bn_into_conv_or_fc(json_str, net_param):
    json_obj, nodes, names, old_num_to_name, inputs = load_json(json_str)
    #json_str = json.dumps(json_obj, indent=4)
    name_to_num = dict([(v, k) for k, v in old_num_to_name.iteritems()])

    bn_name_list = []  # for store the bn_name
    conv_name_list = []  # for store the conv_name

    for i in range(len(json_obj['nodes'])):
        # seach batch-norm and conv(fc)
        if json_obj['nodes'][i]['op'] == "BatchNorm":
            may_conv_index = json_obj['nodes'][i]['inputs'][0][0]

            # search conv or fc before the batchnorm
            if json_obj['nodes'][may_conv_index]['op'] in ["Convolution", "FullyConnected"]:
                bn_name_list.append(json_obj['nodes'][i]['name'])
                conv_name_list.append(
                    json_obj['nodes'][may_conv_index]['name'])

    if len(bn_name_list) != len(conv_name_list) or len(bn_name_list) <= 0:
        print "error, len(bn_name_list) should be equal len(conv_name_list)"
        exit()

    for i in range(len(bn_name_list)):
        print i
        json_obj, nodes, names, old_num_to_name, inputs = load_json(json_str)
        name_to_num = dict([(v, k) for k, v in old_num_to_name.iteritems()])

        # bn_name,bn-eps,bn-fixgamma
        bn_index = name_to_num[bn_name_list[i]]
        bn_name = json_obj['nodes'][bn_index]['name']
        bn_eps = float(json_obj['nodes'][bn_index]['param']['eps'])
        bn_fixgamma = bool(json_obj['nodes'][bn_index]['param']['fix_gamma'])

        # conv_name,no_bias
        conv_index = name_to_num[conv_name_list[i]]
        conv_name = json_obj['nodes'][conv_index]['name']
        conv_no_bias = bool(json_obj['nodes'][conv_index]['param']['no_bias'])

        # use merge_bn_conv_after_bn
        net_param = copy.deepcopy(merge_bn_conv_after(net_param=net_param, conv_name=conv_name,
                                                      bn_name=bn_name, fix_gamma=bn_fixgamma, no_bias=conv_no_bias, eps=bn_eps))
        json_str = copy.deepcopy(merge_bn_conv_after_bn_json(
            json_str=json_str, conv_name=conv_name, bn_name=bn_name, fix_gamma=bn_fixgamma, no_bias=conv_no_bias, eps=bn_eps))

    return json_str, net_param


def load_json(json_str):
    #json_obj = json.load(json_file) # dict contain "nodes arg_nodes, heads"
    json_obj = json.loads(json_str)  # dict contain "nodes arg_nodes, heads"
    nodes = json_obj['nodes']  # a list,lens = num of layers
    names = [node['name'] for node in nodes]  # names
    old_num_to_name = dict(enumerate(names))  # dict
    name_to_num = dict([(v, k) for k, v in old_num_to_name.iteritems()])
    inputs = [node['inputs'] for node in nodes]
    return json_obj, nodes, names, old_num_to_name, inputs


def merge_bn_conv_after_bn_json(json_str, conv_name, bn_name, fix_gamma=False, no_bias=False, eps=0.001):
    json_obj, nodes, names, old_num_to_name, inputs = load_json(json_str)
    name_to_num = dict([(v, k) for k, v in old_num_to_name.iteritems()])
    # cal the conv and bn index
    conv_index = name_to_num[conv_name]
    bn_index = name_to_num[bn_name]
    for i in range(len(json_obj['nodes'])):
        if len(json_obj['nodes'][i]['inputs']) <= 0:
            continue  # when inputs =[]

        # change bn_node to conv_node
        input_list = json_obj['nodes'][i]['inputs']
        for j in range(len(input_list)):
            if input_list[j][0] == bn_index:
                input_list[j][0] = conv_index
            else:
                pass
        json_obj['nodes'][i]['inputs'] = input_list

        # for change bn-layer to a param not op
        if json_obj['nodes'][i]['name'] == bn_name:
            json_obj['nodes'][i] = copy.deepcopy(json_obj['nodes'][i-1])
            json_obj['nodes'][i]['name'] = bn_name

        # change_name
    if no_bias == True:
        #            print json_obj['nodes'][int(bn_index)-1]['name']
        json_obj['nodes'][int(bn_index)-1]['name'] = conv_name + '_bias'
#           print json_obj['nodes'][int(bn_index)-1]['name']
        json_obj['nodes'][conv_index]['param']['no_bias'] = "False"
        list_add = []
        list_add.append(int(bn_index)-1)
        #list_add.append(int(bn_index))
        list_add.append(0)
        json_obj['nodes'][conv_index]['inputs'].append(list_add)

        # change bn_beta_name to conv_bias
        json_obj['nodes'][int(bn_index)-1]['name'] = conv_name + '_bias'

    # return json_obj
    # return json_str
    return json.dumps(json_obj, indent=4)


# merge conv and after bn
def merge_bn_conv_after(net_param, conv_name, bn_name, fix_gamma=False, no_bias=False, eps=0.001):
    gamma = net_param['arg:' + bn_name + '_gamma'].asnumpy()  # scale gamma
    if fix_gamma == True:  # fix_gamma = true
        gamma *= 0
        gamma += 1
    beta = net_param['arg:' + bn_name + '_beta'].asnumpy()  # scale beta
    mov_mean = net_param['aux:' + bn_name +
                         '_moving_mean'].asnumpy()  # bn-mean
    mov_var = net_param['aux:' + bn_name + '_moving_var'].asnumpy()  # bn var
    mov_std = np.sqrt(mov_var + eps)  # calulate the std from var

    # conv_weights and conv_bias before merge
    part_0_conv_weight = net_param['arg:' + conv_name + '_weight'].asnumpy()

    output_channel = part_0_conv_weight.shape[0]  # output_channel
    pdb.set_trace()
    if no_bias == True:  # fill the bias to zero , it is no use has_something wrong
        # update the conv_bias and conv_weights
        part_0_conv_bias = np.zeros((output_channel,), dtype=np.float64)
        #pdb.set_trace()
        # shape[0] is output_channel_num, weight.shape = [out,in,kernel,kernel]
        for i in range(output_channel):
            # update conv_weight
            part_0_conv_weight[i, :, :, :] *= float(gamma[i]/mov_std[i])
     #       part_0_conv_bias[i] *= float(gamma[i]/mov_std[i]) # update conv_bias
            # update conv_bias
            part_0_conv_bias[i] += beta[i] - \
                float(gamma[i]*mov_mean[i]/mov_std[i])
        #pdb.set_trace()

    else:
        # update the conv_bias and conv_weights
        part_0_conv_bias = net_param['arg:' + conv_name + '_bias'].asnumpy()
        # shape[0] is output_channel_num, weight.shape = [out,in,kernel,kernel]
        for i in range(output_channel):
            # update conv_weight
            part_0_conv_weight[i, :, :, :] *= float(gamma[i]/mov_std[i])
            # update conv_bias
            part_0_conv_bias[i] *= float(gamma[i]/mov_std[i])
            # update conv_bias
            part_0_conv_bias[i] += beta[i] - \
                float(gamma[i]*mov_mean[i]/mov_std[i])

    # update the net_param
    net_param['arg:' + conv_name + '_weight'] = mx.nd.array(part_0_conv_weight)
    if no_bias == True:
        #net_param['arg:' + bn_name + '_bias'] = mx.nd.array(part_0_conv_bias)
        net_param['arg:' + conv_name + '_bias'] = mx.nd.array(part_0_conv_bias)
        #pdb.set_trace()
    else:
        net_param['arg:' + conv_name + '_bias'] = mx.nd.array(part_0_conv_bias)
    #print net_param.keys()
    return net_param


# input_mx_model + input_mx_epoch = resnet/base-symbol.json and resnet/base-14-9999.params
input_param = sys.argv[1]  # such as resnet/base-14-9999.params
input_json = file(sys.argv[2])  # such as resnet/base-14.json
net_param = mx.nd.load(input_param)
new_json_str, new_param = merge_bn_into_conv_or_fc(
    input_json.read(), net_param)

#new_json = merge_bn_conv_after_bn_json(json_file = input_json, bn_name="part_0_bn_conv1", conv_name = "part_0_conv",fix_gamma = True, no_bias = False, eps=0.001)
#net_param = merge_bn_conv_after(net_param = net_param, bn_name="part_0_bn_conv1", conv_name = "part_0_conv",fix_gamma = True, no_bias = False, eps=0.001)
#net_param = merge_bn_conv_after(net_param = net_param, bn_name="part_0_bn0", conv_name = "part_0_conv0",fix_gamma = True, no_bias = True ) # for resnet_divide4
#new_json_str = json.dumps(new_json,indent=4)

open((sys.argv[2]).replace(".json", "_change.json"), "w").write(new_json_str)
mx.nd.save(input_param.replace(".params", "_change.params"), new_param)
