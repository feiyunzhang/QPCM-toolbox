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
                         '/m/061hd_', '/m/06m11', '/m/03120', '/m/01kb5b', '/m/0120dh', '/m/0dv5r', '/m/0jbk', '/m/0174n1', '/m/09f_2', '/m/01xq0k1', '/m/03jm5', '/m/02g30s', '/m/05z6w', '/m/01jfm_', '/m/076lb9', '/m/0gj37', '/m/0k0pj', '/m/0kpqd', '/m/0l14j_', '/m/0cyf8', '/m/0174k2', '/m/0dq75', '/m/076bq', '/m/07crc', '/m/0d8zb', '/m/0fszt', '/m/0k1tl', '/m/020kz', '/m/09728', '/m/07j7r', '/m/0fbdv', '/m/03ssj5', '/m/03qrc', '/m/02dl1y', '/m/01k6s3', '/m/02jfl0', '/m/01krhy', '/m/04kkgm', '/m/0ft9s', '/m/0d_2m', '/m/0czz2', '/m/0f4s2w', '/m/07dd4', '/m/0cgh4', '/m/03bbps', '/m/02pjr4', '/m/04p0qw', '/m/02pdsw', '/m/01yx86', '/m/09dzg', '/m/0hkxq', '/m/07dm6', '/m/054_l', '/m/01dws', '/m/027pcv', '/m/01d40f', '/m/02rgn06', '/m/0342h', '/m/06bt6', '/m/0323sq', '/m/02zvsm', '/m/02fq_6', '/m/01lrl', '/m/0k4j', '/m/04h7h', '/m/07xyvk', '/m/03y6mg', '/m/07r04', '/m/03__z0', '/m/019w40', '/m/09j5n', '/m/0cvnqh', '/m/01llwg', '/m/0c9ph5', '/m/015x5n', '/m/0gd2v', '/m/04v6l4', '/m/02jz0l', '/m/0dj6p', '/m/04ctx', '/m/080hkjn', '/m/01c648', '/m/01j61q', '/m/012n7d', '/m/025nd', '/m/09csl', '/m/01lcw4', '/m/0h8n5zk', '/m/0633h', '/m/01fdzj', '/m/01226z', '/m/0mw_6', '/m/04hgtk', '/m/02pv19', '/m/09qck', '/m/063rgb', '/m/0lt4_', '/m/0270h', '/m/01h3n', '/m/01mzpv', '/m/04169hn', '/m/0fm3zh', '/m/0d20w4', '/m/0_cp5', '/m/01dy8n', '/m/03m5k', '/m/03dnzn', '/m/0h8mzrc', '/m/0h8mhzd', '/m/03d443', '/m/01gllr', '/m/0642b4', '/m/09b5t', '/m/04yx4', '/m/01f8m5', '/m/015x4r', '/m/01j51', '/m/02zt3', '/m/03tw93', '/m/01jfsr', '/m/04ylt', '/m/0bt_c3', '/m/0cmx8', '/m/0hqkz', '/m/071qp', '/m/0cyhj_', '/m/01xygc', '/m/0420v5', '/m/0898b', '/m/01knjb', '/m/0199g', '/m/03c7gz', '/m/02x984l', '/m/04zwwv', '/m/04bcr3', '/m/0gv1x', '/m/01nq26', '/m/02s195', '/m/083kb', '/m/06nrc', '/m/0jyfg', '/m/0nybt', '/m/0176mf', '/m/01rzcn', '/m/0d4v4', '/m/03bk1', '/m/096mb', '/m/0h9mv', '/m/07yv9', '/m/0ph39', '/m/01rkbr', '/m/0gjbg72', '/m/06z37_', '/m/01m4t', '/m/035r7c', '/m/019jd', '/m/02tsc9', '/m/015wgc', '/m/0c06p', '/m/01dwwc', '/m/034c16', '/m/0242l', '/m/02lbcq', '/m/03nfch', '/m/03bt1vf', '/m/01lynh', '/m/03q5t', '/m/0fqt361', '/m/01bjv', '/m/01s55n', '/m/0283dt1', '/m/01z1kdw', '/m/016m2d', '/m/02dgv', '/m/07y_7', '/m/01_5g', '/m/06_72j', '/m/0ftb8', '/m/0c29q', '/m/0jg57', '/m/02l8p9', '/m/078jl', '/m/0llzx', '/m/0dbvp', '/m/09ct_', '/m/0dkzw', '/m/02p5f1q', '/m/0fx9l', '/m/01b9xk', '/m/0b3fp9', '/m/0h8n27j', '/m/0h8n6f9', '/m/01599', '/m/017ftj', '/m/044r5d', '/m/01dwsz', '/m/0cdl1', '/m/07gql', '/m/0hdln', '/m/0zvk5', '/m/012w5l', '/m/021sj1', '/m/0bh9flk', '/m/09gtd', '/m/0jwn_', '/m/02wv6h6', '/m/02wv84t', '/m/021mn', '/m/018p4k', '/m/06j2d', '/m/033cnk', '/m/01j3zr', '/m/03fwl', '/m/058qzx', '/m/06_fw', '/m/02x8cch', '/m/04g2r', '/m/01b638', '/m/099ssp', '/m/071p9', '/m/01gkx_', '/m/0b_rs', '/m/03v5tg', '/m/01j5ks', '/m/026t6', '/m/0_k2', '/m/039xj_', '/m/01b7fy', '/m/0220r2', '/m/015p6', '/m/0fly7', '/m/07c52', '/m/0n28_', '/m/0hg7b', '/m/019dx1', '/m/04vv5k', '/m/020jm', '/m/047v4b', '/m/01xs3r', '/m/03kt2w', '/m/03q69', '/m/01dxs', '/m/01h8tj', '/m/0dt3t', '/m/0cjq5', '/m/0h8lkj8', '/m/0271t', '/m/03q5c7', '/m/0fj52s', '/m/03vt0', '/m/01x3z', '/m/0d5gx', '/m/0h8my_4', '/m/03ldnb', '/m/0cjs7', '/m/0449p', '/m/04szw', '/m/07jdr', '/m/01yrx', '/m/06c54', '/m/04h8sr', '/m/050k8', '/m/0pg52', '/m/02f9f_', '/m/054fyh', '/m/09k_b', '/m/03xxp', '/m/0jly1', '/m/06k2mb', '/m/04yqq2', '/m/0bwd_0j', '/m/02h19r', '/m/02zn6n', '/m/07c6l', '/m/05zsy', '/m/025dyy', '/m/07j87', '/m/09ld4', '/m/01vbnl', '/m/0dzct', '/m/03fp41', '/m/0h2r6', '/m/0by6g', '/m/0cxn2', '/m/04tn4x', '/m/0f6wt', '/m/05n4y', '/m/0gxl3', '/m/02d9qx', '/m/04m9y', '/m/05z55', '/m/01x3jk', '/m/0h8l4fh', '/m/031b6r', '/m/01tcjp', '/m/01f91_', '/m/02522', '/m/0319l', '/m/0c_jw', '/m/0l515', '/m/0306r', '/m/0crjs', '/m/0ch_cf', '/m/02xwb', '/m/01r546', '/m/03rszm', '/m/0388q', '/m/03m3pdh', '/m/03k3r', '/m/0hf58v5', '/m/01y9k5', '/m/05441v', '/m/03p3bw', '/m/0175cv', '/m/0cmf2', '/m/0ccs93', '/m/02d1br', '/m/0gjkl', '/m/0jqgx', '/m/0h99cwc', '/m/047j0r', '/m/0k5j', '/m/0h8n6ft', '/m/0gm28', '/m/0130jx', '/m/04rmv', '/m/081qc', '/m/0qmmr', '/m/03fj2', '/m/040b_t', '/m/02y6n', '/m/0fqfqc', '/m/030610', '/m/07kng9', '/m/029b3', '/m/0fbw6', '/m/07qxg_', '/m/068zj', '/m/01g317', '/m/01bfm9', '/m/02068x', '/m/0fz0h', '/m/0jy4k', '/m/05kyg_', '/m/01prls', '/m/01h44', '/m/08pbxl', '/m/02gzp', '/m/04brg2', '/m/031n1', '/m/02jvh9', '/m/046dlr', '/m/0h8ntjv', '/m/0k65p', '/m/011k07', '/m/03grzl', '/m/06y5r', '/m/061_f', '/m/01cmb2', '/m/01mqdt', '/m/05r655', '/m/02p3w7d', '/m/029tx', '/m/04m6gz', '/m/015h_t', '/m/06pcq', '/m/01bms0', '/m/07fbm7', '/m/09tvcd', '/m/06nwz', '/m/0dv9c', '/m/083wq', '/m/0gd36', '/m/0138tl', '/m/07clx', '/m/05ctyq', '/m/0bjyj5', '/m/0dbzx', '/m/02ctlc', '/m/0fp6w', '/m/0djtd', '/m/0167gd', '/m/078n6m', '/m/0152hh', '/m/04gth', '/m/0ll1f78', '/m/0cffdh', '/m/025rp__', '/m/02_n6y', '/m/0wdt60w', '/m/0cydv', '/m/01n5jq', '/m/09rvcxw', '/m/013y1f', '/m/06ncr', '/m/015qff', '/m/024g6', '/m/05gqfk', '/m/0dv77', '/m/052sf', '/m/0cdn1', '/m/03jbxj', '/m/0cyfs', '/m/0kmg4', '/m/02cvgx', '/m/09kx5', '/m/057cc', '/m/02pkr5', '/m/057p5t', '/m/03g8mr', '/m/0frqm', '/m/03m3vtv', '/m/0584n8', '/m/014y4n', '/m/01g3x7', '/m/07cx4', '/m/07bgp', '/m/032b3c', '/m/01bl7v', '/m/0663v', '/m/0cn6p', '/m/02rdsp', '/m/02crq1', '/m/01xqw', '/m/0cnyhnx', '/m/01x_v', '/m/018xm', '/m/09ddx', '/m/084zz', '/m/01n4qj', '/m/07cmd', '/m/04_sv', '/m/0mkg', '/m/09d5_', '/m/0c568', '/m/02wbtzl', '/m/05bm6', '/m/01lsmm', '/m/0dftk', '/m/0dtln', '/m/0nl46', '/m/05r5c', '/m/06msq', '/m/0cd4d', '/m/05kms', '/m/02jnhm', '/m/0fldg', '/m/073bxn', '/m/029bxz', '/m/020lf', '/m/01btn', '/m/02vqfm', '/m/06__v', '/m/043nyj', '/m/0grw1', '/m/03hl4l9', '/m/0hnnb', '/m/04c0y', '/m/0dzf4', '/m/07v9_z', '/m/0f9_l', '/m/0703r8', '/m/01xyhv', '/m/01fh4r', '/m/04dr76w', '/m/0pcr', '/m/03s_tn', '/m/07mhn', '/m/01hrv5', '/m/019h78', '/m/09kmb', '/m/0h23m', '/m/050gv4', '/m/01fb_0', '/m/02w3_ws', '/m/014j1m', '/m/01gmv2', '/m/04y4h8h', '/m/026qbn5', '/m/01m2v', '/m/05_5p_0', '/m/07030', '/m/01s105', '/m/033rq4', '/m/0162_1', '/m/02z51p', '/m/06mf6', '/m/02hj4', '/m/0bt9lr', '/m/08hvt4', '/m/084rd', '/m/01pns0', '/m/014sv8', '/m/079cl', '/m/01940j', '/m/05vtc', '/m/02w3r3', '/m/054xkw', '/m/01bqk0', '/m/09g1w')
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
            filename = self._get_voc_results_file_template().format(cls.split('/')[-1])
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
            filename = self._get_voc_results_file_template().format(cls.split('/')[-1])
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls.split('/')[-1] + '_pr.pkl'), 'w') as f:
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
        self._do_python_eval(output_dir)
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
