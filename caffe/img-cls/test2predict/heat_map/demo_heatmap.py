#coding=utf-8  
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "lib/caffe/python")
import caffe
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time

def parser():
    parser = ArgumentParser('AtLab Label Image!')
    parser.add_argument('--images',dest='im_path',help='Path to the image',
                        default='images/',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='h_and_o_result/',type=str)
    return parser.parse_args()

def norm_feat(feature):
    batchsize, channel_num, fea_height,fea_width = feature.shape
    normed_feature = np.zeros_like(feature)
    for c in xrange(channel_num):
        channel_fea = feature[0,c,:,:]
        channel_mean = np.mean(channel_fea)
        channel_fea = np.square(channel_fea - channel_mean) * channel_fea
        normed_feature[0,c,:,:] = channel_fea
    one_feature = np.mean(normed_feature,axis=1)
    one_feature = np.squeeze(one_feature)
    one_feature = one_feature * 255 / np.max(one_feature)
    return one_feature

def draw_heatmap(fea, save_path, size):
    fig = plt.figure()
    fig.set_size_inches(1. * size[1] / size[0], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(fea, interpolation='bicubic', cmap='jet')
    plt.savefig(save_path, dpi = size[0])
    plt.close()

def draw_weighted_img(img, heatmap, hm_path):
    '''
    Overlaps the heatmap with the original image

    Args：
        img: numpy array, original image
        heatmap: numpy array, the dimensions of heatmap are the same as the dimensions of the image
        hm_path: string, combination image save path
    '''
    overlay = img.copy()
    alpha = 0.3
    cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]),
                (200, 0, 0), -1)  # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0, img)  # 将热度图覆盖到原图
    cv2.imwrite(hm_path, img)
    cv2.waitKey(0)


def center_crop(img, crop_size): 
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]

if __name__ == '__main__':
    args = parser()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net_cls = caffe.Net("lib/models/extract_feature.prototxt", "lib/models/t7_32000.caffemodel", caffe.TEST)
       
    filename = os.listdir(args.im_path)
    filename.sort()

    for im in filename:
        ori_img = cv2.imread(os.path.join(args.im_path, im))
        size = ori_img.shape[0:2]
        img = ori_img.copy()
        img = cv2.resize(img, (225, 225))
        img = img.astype(np.float32, copy=True)
        img -= np.array([[[103.94,116.78,123.68]]])
        img = img * 0.017
        #img = center_crop(img, 225)
        img = img.transpose((2, 0, 1))
        net_cls.blobs['data'].data[...] = img
        out = net_cls.forward() 
        fea = out['block_5_3']
        normed_feat = norm_feat(fea)
        normed_feat = cv2.resize(normed_feat, (size[1], size[0])) # 将feature的shape转换为image的shape

        heatmap_path = './h_result/' + im
        draw_heatmap(normed_feat, heatmap_path, size)
        heatmap = cv2.cvtColor(cv2.imread(heatmap_path), cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (size[1], size[0]))

        # draw weighted image, combination of original pictures and heatmap
        draw_weighted_img(ori_img, heatmap, os.path.join(args.out_path, im))

         



    
