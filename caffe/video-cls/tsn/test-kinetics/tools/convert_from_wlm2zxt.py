import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/cls_cnt)

zxt_label = np.load('../../../../res101GBD-flow-f1_mini_test/score_file/62.88_res101_f1_flow_256_288_png.npz')
wlm_bn_npz = np.load('1.npz')

zxt2wlm = np.loadtxt('label-dict-from-zxt2wlm.txt',str,delimiter='\n')

new_scores = np.zeros_like(wlm_bn_npz['scores'])
print np.shape(new_scores)

for i, score in enumerate(wlm_bn_npz['scores']):
    per_video = np.zeros_like(score)
    for p in range(400):
        per_video[p] = score[int(zxt2wlm[p].split(': ')[1])]
    new_scores[i] = per_video

new_label = zxt_label['labels']
acc = mean_class_accuracy(new_scores, new_label)
print 'Final accuracy {:02f}%'.format(acc * 100)

np.savez('61.50_bn_inception.npz', scores=new_scores, labels=new_label)
