import argparse
import itertools
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_consumers',type=int,default=1)
args = parser.parse_args()

args.score_files=['test29328-new_60.17_flow_tsn-pytorch_inceptionresnetv2_200e_norm.npz',
                  'test29328-new_61.51_flow_tsn-caffe_res152GBD_39w_340x256_norm.npz',
'test29328-new_64.17_flow_tsn-caffe_se-next101_43w_340x256_norm.npz',
'test29328-new_69.29_rgb_tsn_caffe_res269GBD_3n_425x320_29w_norm.npz',
'test29328-new_69.63_rgb_tsn_caffe_res269GBD_3n_384x288_29w_norm.npz',
'test29328-new_71.19_rgb_tsn_caffe_se-next101_26w_425x320_norm.npz',
'test29328-new_71.27_rgb_tsn-pytorch_inceptionv4_norm.npz',
'test29328-new_71.56_rgb_tsn_caffe_se-next101_26w_340x256_norm.npz',
'test29328-new_71.63_rgb_tsn_caffe_se-next101_26w_425x320_norm.npz',
'test29328-new_73.1_rgb_tsn-pytorch_dpn107_80e_norm.npz',
'test29328-new_73.35_rgb_tsn-pytorch_resnext101_64x4d_80e_norm.npz',
'test29328-old_59.64_flow_tsn-pytorch_inceptionv3_80e_norm.npz',
'test29328-old_69.92_rgb_tsn-pytorch_inceptionresnetv2_80e_norm.npz',
'test29328-old_70.43_rgb_tsn_caffe_res152GBD_nk9_340x256_29w_norm.npz',
'test29328-old_70.61_rgb_tsn-pytorch_resnet152_80e_norm.npz',
'test29328-old_71.4_rgb_tsn_caffe_res152GBD_nk6_340x256_26w_norm.npz',
'test29328-old_71.69_rgb_tsn-pytorch_resnext101_64x4d_100e_norm.npz',
'test29328-old_71.93_rgb_tsn_caffe_se-res152_nk5_340x256_28w_norm.npz',
'test29328-old_72.12_rgb_tsn_caffe_res269GBD_nk4_340x256_28w_norm.npz',
'test29328-old_72.15_rgb_tsn-pytorch_inceptionv4_80e_norm.npz',
'test29328-old_72.84_rgb_tsn-pytorch_dpn107_80e_norm.npz',
'test29328-old_73.04_rgb_non_local_baseline_res50_norm.npz',
'test29328-old_73.21_rgb_tsn-pytorch_dpn107-seg7_76e_norm.npz',
'test29328-old_73.80_rgb_i3d-pytorch_i3ddpn107_80e_norm.npz',
'test29328-old_75.40_rgb_non_local_baseline_res101_norm.npz',
'test29328-new_62.26_flow_tsn-caffe_res269GBD_44w_340x256_norm.npz',
'test29328-old_72.22_rgb_tsn-pytorch_inceptionv3_80e_norm.npz',
'test29328-new_63.9_flow_tsn-caffe_se-next101_50w_384x288_norm.npz',
'test29328-old_64.65_flow_tsn-pytorch_dpn107_198e_norm.npz',
'test29328-new_64.19_flow_tsn-caffe_se-next101_50w_340x256_norm.npz',
'test29328-old_77.81_rgb_non_local_res50_norm.npz',
'test29328-new_78.08_rgb_non_local_res50_norm.npz',
'test29328-old_79.03_rgb_non_local_res101_norm.npz',
'test29328-new_79.11_rgb_non_local_res101_norm.npz']

args.score_weights = [1,1,2,2,1,2,4,4]

score_npz_files = [np.load(x) for x in args.score_files]
score_list = [x['scores'] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]

def param_search(score_files, init_score_weights):
    print("preparing data")

    print("starting grid search")
    elem_range = []
    for i in range(len(score_files)-len(init_score_weights)):
        elem_range.append(range(0, 2, 1))

    search_grid = list(itertools.product(*elem_range))
    print("grid size:",len(search_grid))

    pool = Pool(processes=args.num_consumers)
    for current_score_weights in search_grid:
        pool.apply_async(main, args=(current_score_weights,init_score_weights))

    pool.close()
    pool.join()
    #resbrute = scipy.optimize.brute(main, search_grid, args=[score_list,label_list,init_score_weights], finish = None)

def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/(cls_cnt+1e-9))

def top_1_accuracy(scores, labels):
    pred_label = np.argmax(scores, axis=1)
    correct_num = np.sum(pred_label==labels)
    return float(correct_num) / len(labels)

def top_5_accuracy(scores, labels):
    pred_label_sorted = np.argsort(scores,axis=1)
    score_top5 = 0
    for idx,sample_label_sorted in enumerate(pred_label_sorted):
        if labels[idx] in sample_label_sorted[len(sample_label_sorted)-5:]:
            score_top5 += 1
    return float(score_top5) / len(labels)

def main(score_weights,init_score_weights):
    score_weights = list(score_weights) + init_score_weights

    final_scores = np.zeros_like(score_list[0])
    for i, agg_score in enumerate(score_list):
        final_scores += agg_score * score_weights[i]

    acc = top_1_accuracy(final_scores, label_list[0])

    acc_top5 = top_5_accuracy(final_scores, label_list[0])

    final_acc = (acc + acc_top5) / 2.0
    final_err = 1.0 - final_acc
    print('Current weights:{}, Top-1 acc:{:02f}, '
          'Top-5 acc:{}, Final error:{}'.format(score_weights,acc*100,acc_top5*100,final_err*100))
    return final_err

if __name__ == '__main__':
    param_search(args.score_files,args.score_weights)