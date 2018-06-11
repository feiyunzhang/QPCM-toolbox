import argparse
import sys
import numpy as np
sys.path.append('.')
from sklearn.metrics import average_precision_score, confusion_matrix
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
args = parser.parse_args()

def evaluate(submit, target):
    prec = {}
    for k in [1, 5]:
        correct = [1 if line in submit[i][:k] else 0 for i,line in enumerate(target)]
        prec[k] = 100*float(sum(correct)) / float(len(correct))
    score = np.mean(list(prec.values()))
    return prec, score

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/cls_cnt)
def top_1_accuracy(scores, labels):
    score_top5 = 0
    for i in range(len(scores)):
        sort_pre = sorted(enumerate(scores[i]) ,key=lambda z:z[1])
        pred_label = [sort_pre[-j][0] for j in range(1,2)]
        if labels[i] in pred_label:
            score_top5 = score_top5 + 1
    return float(score_top5/float(len(scores)))

def top_5_accuracy(scores, labels):
    score_top5 = 0
    for i in range(len(scores)):
        sort_pre = sorted(enumerate(scores[i]) ,key=lambda z:z[1])
        pred_label = [sort_pre[-j][0] for j in range(1,6)]
        if labels[i] in pred_label:
            score_top5 = score_top5 + 1
    return float(score_top5/float(len(scores)))


score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]


final_scores = np.zeros_like(score_list[0])
weights_num = 0
for i, agg_score in enumerate(score_list):
    final_scores += agg_score * score_weights[i]
    weights_num += score_weights[i]
print 'weights_num:', weights_num

final_scores = final_scores/float(weights_num)

t = np.argsort(final_scores, axis=1)
tt = t[:,-5:][:,::-1]

prec, score = evaluate(tt, label_list[0])
print('Top 1 accuracy: {}'.format(100-prec[1]))
print('Top 5 accuracy: {}'.format(100-prec[5]))
print('Final score: {}'.format(100-score))

'''
acc = top_1_accuracy(final_scores, label_list[0])
print 'Top-1 accuracy {:02f}%'.format(acc * 100)

acc_top5 = top_5_accuracy(final_scores, label_list[0])
print 'Top-5 accuracy {:02f}%'.format(acc_top5 * 100)

final_acc = (acc + acc_top5)/2.0
print 'final_acc accuracy {:02f}%'.format(final_acc * 100)

final_err = 1.0 - final_acc
print 'final_acc error {:02f}%'.format(final_err * 100)
'''
if args.save_scores is not None:
    np.savez(args.save_scores, scores=final_scores, labels=label_list[0])