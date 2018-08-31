import pandas as pd
from xml.etree import  ElementTree
import numpy as np
import cPickle
import sys
import os
import os.path as osp
import sys
from multiprocessing import Pool
from tqdm import tqdm
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_path = './'
add_path(root_path+'lib')

from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper, soft_nms, nms
from datasets import *
from datasets.factory import get_imdb
from utils.cython_bbox import bbox_overlaps


def bbox_voting(cls_dets_after_nms, cls_dets, threshold):
    """
    A nice trick to improve performance durning TESTING.
    Check 'Object detection via a multi-region & semantic segmentation-aware CNN model' for details.
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(cls_dets_after_nms[:, :4], dtype=np.float),
        np.ascontiguousarray(cls_dets[:, :4], dtype=np.float))
    for i in xrange(cls_dets_after_nms.shape[0]):
        candidate_bbox = cls_dets[overlaps[i, :] >= threshold, :]
        cls_dets_after_nms[i, :4] = np.average(candidate_bbox[:, :4], axis=0, weights=candidate_bbox[:, 4])
    return cls_dets_after_nms

def tester(fusion, NUM, bbox_vote=False, max_per_image=400):
    all_boxes = [[[] for _ in range(99999)]
             for _ in range(501)]

    for i in tqdm(range(99999), file=sys.stdout, leave=False, dynamic_ncols=True):
        for j in range(1,501):
            det_boxes = np.vstack((fusion[q][j][i] for q in range(NUM)))
            keep = nms(det_boxes, 0.4)
            det_boxes_after_nms = det_boxes[keep, :]
            if bbox_vote:
                cls_dets_after_vote = bbox_voting(det_boxes_after_nms, det_boxes, threshold=0.5)
                all_boxes[j][i] = cls_dets_after_vote
            else:
                all_boxes[j][i] = det_boxes_after_nms

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, 501)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, 501):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

    '''step-3: save and eval'''
    with open('output/model_all-test-mst-nms0.4-bbox-vote0.5.pkl', 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    '''step-1: load multi-scale pkl'''
    fusion = []
    detection = os.listdir('./pkl/')
    NUM = len(detection)
    for f in tqdm(range(NUM), file=sys.stdout, leave=False, dynamic_ncols=True):
        fusion.append(cPickle.load(open('./pkl/' + detection[f],'rb')))

    '''step-2: merge all pkl in all_boxes'''
    tester(fusion, NUM, bbox_vote=True)
    
