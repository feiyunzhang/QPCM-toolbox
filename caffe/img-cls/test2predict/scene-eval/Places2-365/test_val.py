import numpy as np
import cPickle
import sys
import os
import cv2

if __name__ == '__main__':
    sys.path.insert(0, "caffe_new/python")
    import caffe
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(1) 
    net = caffe.Net("models/Places2-365-CNN.prototxt", "models/Places2-365-CNN.caffemodel", caffe.TEST)
    result = open('val_result.txt','w')
    cls_list = np.loadtxt('models/places365_val.txt', str, delimiter='\n')

    filename = os.listdir('places2_val/')
    filename.sort()

    score_top5 = 0
    score_top1 = 0
    index = 1
    for i in range(len(filename)):
        origimg = cv2.imread('places2_val/' + filename[i])
        img = cv2.resize(origimg, (224, 224))
        img = img.astype(np.float32, copy=False)
        img -= np.array([[[105.448, 113.768, 116.052]]])
        img = img.transpose((2, 0, 1))
        net.blobs['data'].data[...] = img
        out = net.forward() 
        print net.blobs['prob'].data
        score = out['prob'][0] 
        sort_pre = sorted(enumerate(score) ,key=lambda z:z[1])
        label = [sort_pre[-j][0] for j in range(1,6)]
        score = [sort_pre[-j][1] for j in range(1,6)]
        result.write(filename[i] + ' ' + str(label[0]) + ' ' + str(label[1]) + ' ' + str(label[2]) + ' ' + str(label[3]) + ' ' + str(label[4]) + '\n')
        
        if int(cls_list[i].split(' ')[1]) in label: 
            score_top5 = score_top5 + 1
        if int(cls_list[i].split(' ')[1]) == label[0]:
            score_top1 = score_top1 + 1
        index  = index + 1
        print 'Top5-acc:', score_top5/float(index)
        if index % 1000 == 0:
            print index, ' is done!'
    Top5_acc = float(score_top5)/36500.0
    Top1_acc = float(score_top1)/36500.0
    print 'Top5-acc', Top5_acc
    print 'Top1-acc', Top1_acc
    result.close()


    
