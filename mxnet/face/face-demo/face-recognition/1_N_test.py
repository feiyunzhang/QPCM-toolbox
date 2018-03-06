import lib.face_embedding
import argparse
import cv2
import numpy as np
import os
import cPickle

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='lib/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=4, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = lib.face_embedding.FaceModel(args)
feat_pkl_euc = os.path.join('1_N/feature/', 'result_euc_1.pkl')
feat_pkl_cos = os.path.join('1_N/feature/', 'result_cos_1.pkl')
face_path = '1_N/scene-data/'
face_img = os.listdir(face_path)
lib_feature = cPickle.load(open('1_N/feature/facelib_feature.pkl','rb'))
index = 0
result_dict_euc = {}
result_dict_cos = {}
for img_file in face_img:
    index = index + 1
    print index
    img = cv2.imread(face_path + img_file)
    result_list_euc = []
    result_list_cos = []
    if np.shape(img) != ():
        f1 = model.get_feature(img)
        for key in lib_feature.keys():
            f2 = lib_feature[key]
            if np.shape(f1) == () or np.shape(f2) == ():
                dist = 10.0
                sim = -1.0
            else:
                dist = np.sum(np.square(f1-f2))
                sim = np.dot(f1, f2.T)
            result_euc = [dist, key]
            result_cos = [sim, key]
            result_list_euc.append(result_euc)
            result_list_cos.append(result_cos)
    result_list_euc.sort()
    result_list_cos.sort(reverse=True)
    #print result_list_euc[0]
    #print result_list_euc[-1]   
    euc = []
    cos = [] 
    for i in range(100):
        euc.append(result_list_euc[i])
        cos.append(result_list_cos[i])
    result_dict_euc[img_file] = euc
    result_dict_cos[img_file] = cos
    
with open(feat_pkl_euc, 'wb') as f:
        cPickle.dump(result_dict_euc, f, cPickle.HIGHEST_PROTOCOL)

with open(feat_pkl_cos, 'wb') as f:
        cPickle.dump(result_dict_cos, f, cPickle.HIGHEST_PROTOCOL)
