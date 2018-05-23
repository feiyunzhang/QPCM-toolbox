# -*- coding:utf-8 -*-
"""
"""

import os
import sys
import os
import urllib
import mxnet as mx

# 下面这两个函数是用来下载模型的


def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)


def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))


def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./data/Terror-Classify-V0.3-train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True)
    val = mx.io.ImageRecordIter(
        path_imgrec='./data/Terror-Classify-V0.3-val.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False)
    return (train, val)


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(
        data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


import logging


def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    #devs = [mx.gpu(i) for i in range(num_gpus)]
    devs = [mx.gpu(i) for i in [0, 5, 6, 7]]
    model_prefix = './models/outputModels/Terror-Classify-V0.3_resnet_50_t1'
    checkpoint = mx.callback.do_checkpoint(model_prefix, 10)
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=200,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            epoch_end_callback=checkpoint,
            batch_end_callback=mx.callback.Speedometer(batch_size, 10),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(
                rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)


def main():
    # 下面这句话 是从网上下载 resnet 50 模型的
    # 由于已经下载过了，就直接加载了。
    # get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        './models/preModels/resnet-50', 0)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    num_classes = 48  # 由于添加了两个类别
    batch_per_gpu = 8  # be careful if 16 will be out of memory
    num_gpus = 4
    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
    batch_size = batch_per_gpu * num_gpus
    (train, val) = get_iterators(batch_size)
    mod_score = fit(new_sym, new_args, aux_params,
                    train, val, batch_size, num_gpus)
    pass


if __name__ == '__main__':
    main()
