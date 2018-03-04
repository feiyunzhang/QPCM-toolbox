#!/usr/bin/env sh
./caffe_new/build/tools/caffe train \
--solver=solver.prototxt \
--weights=models/se_resnet_50_v1.caffemodel \
--gpu=0,1
