# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import h5py

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        #self._classes = ('__background__', # always index 0
        #                 'aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')
        self._classes = ('__background__',#always index 0
                         '/m/01g317', '/m/09j2d', '/m/04yx4', '/m/0dzct', '/m/07j7r', '/m/05s2s', '/m/03bt1vf', '/m/07yv9', '/m/0cgh4', '/m/01prls', '/m/09j5n', '/m/05r655', '/m/0jbk', '/m/0k4j', '/m/02wbm', '/m/083wq', '/m/0c9ph5', '/m/0c_jw', '/m/0d4v4', '/m/03jm5', '/m/01bl7v', '/m/0463sg', '/m/04bcr3', '/m/0jyfg', '/m/01xyhv', '/m/08dz3q', '/m/015p6', '/m/05y5lj', '/m/01d40f', '/m/0bt9lr', '/m/01lrl', '/m/02p0tk3', '/m/0fly7', '/m/04szw', '/m/0271t', '/m/019jd', '/m/03q69', '/m/0h9mv', '/m/04hgtk', '/m/01yrx', '/m/01rzcn', '/m/01mzpv', '/m/0199g', '/m/01fdzj', '/m/04rky', '/m/079cl', '/m/0dzf4', '/m/0138tl', '/m/06msq', '/m/03xxp', '/m/0hg7b', '/m/01n5jq', '/m/03vt0', '/m/0342h', '/m/0k0pj', '/m/02dl1y', '/m/04brg2', '/m/02dgv', '/m/01bqk0', '/m/017ftj', '/m/052lwg6', '/m/014sv8', '/m/0270h', '/m/0283dt1', '/m/0k5j', '/m/0cmf2', '/m/07jdr', '/m/032b3c', '/m/033rq4', '/m/0k65p', '/m/01ww8y', '/m/0zvk5', '/m/07mhn', '/m/04dr76w', '/m/03fp41', '/m/03k3r', '/m/01y9k5', '/m/0cdl1', '/m/0f4s2w', '/m/02xwb', '/m/035r7c', '/m/0bt_c3', '/m/01_bhs', '/m/01599', '/m/03120', '/m/026t6', '/m/01bjv', '/m/07r04', '/m/018xm', '/m/01rkbr', '/m/0fm3zh', '/m/02_n6y', '/m/04_sv', '/m/06z37_', '/m/01bfm9', '/m/0h8mhzd', '/m/0d_2m', '/m/0gjbg72', '/m/01n4qj', '/m/0ch_cf', '/m/06m11', '/m/01jfm_', '/m/02crq1', '/m/083kb', '/m/01c648', '/m/09tvcd', '/m/0h2r6', '/m/081qc', '/m/09ddx', '/m/03p3bw', '/m/0cyf8', '/m/0b_rs', '/m/039xj_', '/m/021sj1', '/m/0dv5r', '/m/01lynh', '/m/06bt6', '/m/01226z', '/m/0fszt', '/m/050k8', '/m/02wbtzl', '/m/02p5f1q', '/m/025nd', '/m/02522', '/m/09ct_', '/m/0cvnqh', '/m/0d5gx', '/m/01xygc', '/m/04m6gz', '/m/01gkx_', '/m/01s105', '/m/01j61q', '/m/0hnnb', '/m/01j51', '/m/01knjb', '/m/03__z0', '/m/01m2v', '/m/0167gd', '/m/0284d', '/m/03ssj5', '/m/02fq_6', '/m/06nwz', '/m/0220r2', '/m/01mqdt', '/m/0268lbt', '/m/07c52', '/m/0grw1', '/m/01h3n', '/m/078n6m', '/m/01xq0k1', '/m/0gd2v', '/m/0dbvp', '/m/03rszm', '/m/0h8n5zk', '/m/019dx1', '/m/03hl4l9', '/m/0b3fp9', '/m/02rdsp', '/m/0hf58v5', '/m/04h7h', '/m/024g6', '/m/0cffdh', '/m/03q5c7', '/m/014y4n', '/m/01yx86', '/m/020jm', '/m/025dyy', '/m/01llwg', '/m/08pbxl', '/m/02d9qx', '/m/02pkr5', '/m/0h99cwc', '/m/050gv4', '/m/02vqfm', '/m/09kx5', '/m/019w40', '/m/09dzg', '/m/07k1x', '/m/080hkjn', '/m/07qxg_', '/m/0ph39', '/m/018p4k', '/m/02h19r', '/m/015h_t', '/m/0fqfqc', '/m/025rp__', '/m/01x3z', '/m/0crjs', '/m/0l515', '/m/015qff', '/m/09kmb', '/m/09728', '/m/071qp', '/m/02s195', '/m/06c54', '/m/01xqw', '/m/05zsy', '/m/0bwd_0j', '/m/04m9y', '/m/052sf', '/m/03grzl', '/m/01z1kdw', '/m/02wv6h6', '/m/016m2d', '/m/0dtln', '/m/057cc', '/m/06k2mb', '/m/0f6wt', '/m/0cxn2', '/m/02jvh9', '/m/0gjkl', '/m/01b638', '/m/071p9', '/m/0pg52', '/m/0ftb8', '/m/0hnyx', '/m/02jz0l', '/m/04kkgm', '/m/0174n1', '/m/0gv1x', '/m/09csl', '/m/02jnhm', '/m/099ssp', '/m/03nfch', '/m/07y_7', '/m/05z6w', '/m/03m3pdh', '/m/09ld4', '/m/09b5t', '/m/054xkw', '/m/0130jx', '/m/07fbm7', '/m/01dws', '/m/01tcjp', '/m/0dftk', '/m/0c06p', '/m/034c16', '/m/09d5_', '/m/03hlz0c', '/m/0ft9s', '/m/011k07', '/m/054_l', '/m/0jqgx', '/m/0663v', '/m/0242l', '/m/014trl', '/m/05r5c', '/m/07j87', '/m/05kyg_', '/m/0kmg4', '/m/07cmd', '/m/0dv77', '/m/096mb', '/m/01gmv2', '/m/07bgp', '/m/0cmx8', '/m/029tx', '/m/073bxn', '/m/0bh9flk', '/m/06mf6', '/m/06_fw', '/m/078jl', '/m/0fbdv', '/m/0h23m', '/m/014j1m', '/m/03fwl', '/m/02y6n', '/m/06c7f7', '/m/026qbn5', '/m/0cdn1', '/m/07clx', '/m/07cx4', '/m/03g8mr', '/m/0cnyhnx', '/m/01b7fy', '/m/04gth', '/m/0cyfs', '/m/021mn', '/m/07dm6', '/m/0k1tl', '/m/0dv9c', '/m/0dt3t', '/m/04yqq2', '/m/01cmb2', '/m/0gd36', '/m/033cnk', '/m/06ncr', '/m/03bk1', '/m/0bjyj5', '/m/06__v', '/m/0qmmr', '/m/01xgg_', '/m/0czz2', '/m/02l8p9', '/m/09g1w', '/m/0ll1f78', '/m/0cyhj_', '/m/0642b4', '/m/0h8mzrc', '/m/068zj', '/m/02z51p', '/m/0h8nr_l', '/m/0388q', '/m/02hj4', '/m/01jfsr', '/m/07gql', '/m/0h8my_4', '/m/0n28_', '/m/0120dh', '/m/020kz', '/m/0mkg', '/m/03c7gz', '/m/09k_b', '/m/031n1', '/m/04rmv', '/m/084rd', '/m/02rgn06', '/m/05z55', '/m/01r546', '/m/09qck', '/m/012w5l', '/m/01940j', '/m/09f_2', '/m/02p3w7d', '/m/057p5t', '/m/0d8zb', '/m/01nq26', '/m/01x_v', '/m/05gqfk', '/m/0cydv', '/m/07030', '/m/084zz', '/m/0c29q', '/m/02zn6n', '/m/03tw93', '/m/0fqt361', '/m/0f9_l', '/m/0gm28', '/m/09rvcxw', '/m/01nkt', '/m/04p0qw', '/m/03hj559', '/m/07pj7bq', '/m/04ctx', '/m/0703r8', '/m/03qrc', '/m/020lf', '/m/0by6g', '/m/01fh4r', '/m/07c6l', '/m/03bj1', '/m/0898b', '/m/02x984l', '/m/0fj52s', '/m/0cd4d', '/m/02068x', '/m/01vbnl', '/m/0449p', '/m/0gj37', '/m/0nl46', '/m/0152hh', '/m/03dnzn', '/m/05_5p_0', '/m/02jfl0', '/m/01dxs', '/m/0cjq5', '/m/040b_t', '/m/0_cp5', '/m/0gxl3', '/m/029bxz', '/m/02zt3', '/m/03d443', '/m/0306r', '/m/0h8l4fh', '/m/0633h', '/m/01s55n', '/m/0hkxq', '/m/0cn6p', '/m/0dbzx', '/m/01dy8n', '/m/01h8tj', '/m/03s_tn', '/m/01xs3r', '/m/031b6r', '/m/06j2d', '/m/0hqkz', '/m/01_5g', '/m/02zvsm', '/m/0kpqd', '/m/015x4r', '/m/061hd_', '/m/04ylt', '/m/02wv84t', '/m/04y4h8h', '/m/01gllr', '/m/0pcr', '/m/0jy4k', '/m/09f20', '/m/0mcx2', '/m/04c0y', '/m/0_k2', '/m/0jg57', '/m/03fj2', '/m/03ldnb', '/m/06nrc', '/m/01btn', '/m/05vtc', '/m/08hvt4', '/m/0fx9l', '/m/01h44', '/m/05n4y', '/m/0jly1', '/m/06y5r', '/m/05ctyq', '/m/0fp6w', '/m/0d4w1', '/m/02pv19', '/m/07crc', '/m/01dwwc', '/m/01b9xk', '/m/013y1f', '/m/0m53l', '/m/0174k2', '/m/01dwsz', '/m/04vv5k', '/m/04cp_', '/m/0fz0h', '/m/0llzx', '/m/0319l', '/m/04v6l4', '/m/0dkzw', '/m/027pcv', '/m/0323sq', '/m/054fyh', '/m/01pns0', '/m/012n7d', '/m/044r5d', '/m/01krhy', '/m/0dq75', '/m/0176mf', '/m/0h8lkj8', '/m/04tn4x', '/m/06l9r', '/m/0cjs7', '/m/01lsmm', '/m/01lcw4', '/m/047j0r', '/m/01fb_0', '/m/04169hn', '/m/076bq', '/m/0hdln', '/m/01g3x7', '/m/0l3ms', '/m/058qzx', '/m/0h8n6ft', '/m/018j2', '/m/0l14j_', '/m/0wdt60w', '/m/02gzp', '/m/0h8n6f9', '/m/0fbw6', '/m/07kng9', '/m/0dj6p', '/m/06pcq', '/m/061_f', '/m/04g2r', '/m/0jwn_', '/m/02f9f_', '/m/01f8m5', '/m/01m4t', '/m/0cl4p', '/m/07xyvk', '/m/084hf', '/m/03v5tg', '/m/0qjjc', '/m/015x5n', '/m/0ccs93', '/m/0nybt', '/m/02vkqh8', '/m/09gtd', '/m/019h78', '/m/015wgc', '/m/01x3jk', '/m/01j3zr', '/m/0c568', '/m/02pdsw', '/m/029b3', '/m/03q5t', '/m/0p833', '/m/01d380', '/m/024d2', '/m/0mw_6', '/m/01f91_', '/m/02g30s', '/m/01hrv5', '/m/03m5k', '/m/0162_1', '/m/063rgb', '/m/06_72j', '/m/046dlr', '/m/047v4b', '/m/04zpv', '/m/043nyj', '/m/03bbps', '/m/02w3r3', '/m/02pjr4', '/m/0755b', '/m/02lbcq', '/m/0fldg', '/m/012074', '/m/035vxb', '/m/02w3_ws', '/m/0f6nr', '/m/0420v5', '/m/0frqm', '/m/03txqz', '/m/0271qf7', '/m/074d1', '/m/08p92x', '/m/01j4z9', '/m/0kpt_', '/m/0h8n27j', '/m/03y6mg', '/m/04h8sr', '/m/065h6l', '/m/02tsc9', '/m/012ysf', '/m/0ky7b', '/m/02wg_p', '/m/0584n8', '/m/03kt2w', '/m/05kms', '/m/030610', '/m/0lt4_', '/m/076lb9', '/m/02ctlc', '/m/02x8cch', '/m/09gys', '/m/03jbxj', '/m/012xff', '/m/0h8kx63', '/m/073g6', '/m/02cvgx', '/m/027rl48', '/m/01kb5b')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        #if int(self._year) == 2007 or self._image_set != 'test':
        if self._image_set != 'val' and self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        #raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        f = h5py.File(filename,'r')
        box_list = []
        for i in xrange(len(f['boxes'])):
            _box = f[f['boxes'][i][0]].value
            box = _box[:,:].T
            box_list.append(box[:, (1, 0, 3, 2)] - 1)
        #for i in xrange(raw_data.shape[0]):
        #    boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        #    keep = ds_utils.unique_boxes(boxes)
        #    boxes = boxes[keep, :]
        #    keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
        #    boxes = boxes[keep, :]
        #    box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # img_size, width, and height
        img_size = data.getElementsByTagName('size')
        width = float(get_data_from_tag(img_size[0], 'width'))
        height = float(get_data_from_tag(img_size[0], 'height'))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            #x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            #y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            #x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            #y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            
            # 0-based, ensure xmax and ymax do NOT big than img_size
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        print 'comp_id:', self._get_comp_id()
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                     # change to 0-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        #self._do_python_eval(output_dir)
        # do NOT do eval
        #if self.config['matlab_eval']:
        #    self._do_matlab_eval(output_dir)
        #if self.config['cleanup']:
        #    for cls in self._classes:
        #        if cls == '__background__':
        #            continue
        #        filename = self._get_voc_results_file_template().format(cls)
        #        os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
