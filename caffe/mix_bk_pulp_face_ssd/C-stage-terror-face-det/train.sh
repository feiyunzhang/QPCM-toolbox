#!/usr/bin/env sh
caffe-ssd/build/tools/caffe train --solver=models/det_solver.prototxt --weights=models/B-stage-pulp-cls_iter_60000.caffemodel --gpu=0,1,2,3

