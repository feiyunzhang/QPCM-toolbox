"""
convert bbox_pred_delta in previous stage to proposals in next stage.
Added by zhbli
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from bbox.bbox_transform import bbox_pred, clip_boxes
from rpn.generate_anchor import generate_anchors
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

DEBUG = False


class DecodeBboxOperator(mx.operator.CustomOp):
    def __init__(self):
        super(DecodeBboxOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        :param is_train:
        :param req:
        :param in_data: in_data[0] rois: (128, 5) First col are all 0's. True coordinate.
                        in_data[1] bbox_deltas: (128, 8)
                        in_data[2] im_info: im.shape = (im_info[0], im_info[1])
        :param out_data:
        :param aux:
        :return:
        '''
        rois = in_data[0].asnumpy()[:, 1:] # (128, 4) Move 0's in first col.
        bbox_deltas = in_data[1].asnumpy()
        im_info = in_data[2].asnumpy()[0, :]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(rois, bbox_deltas)
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2]) # (128, 8) First 4 cols: background, last 4 cols: object
        proposals = proposals[:, 4:] # (128, 4)
        zeros = np.zeros((proposals.shape[0], 1), dtype=proposals.dtype)
        proposals = np.hstack((zeros, proposals))
        self.assign(out_data[0], req[0], proposals)

        if DEBUG:
            print proposals

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register("decode_bbox")
class DecodeBboxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DecodeBboxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['rois', 'bbox_deltas', 'im_info']

    def list_outputs(self):
        return['proposals']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        assert rois_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in reg'
        im_info_shape = in_shape[2]
        proposals_shape = (in_shape[0][0], long(5))

        return [rois_shape, bbox_pred_shape, im_info_shape], [proposals_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DecodeBboxOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
