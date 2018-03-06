import lib.face_embedding
import argparse
import cv2
import numpy as np
import os
import cPickle

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='lib/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

feat_pkl = os.path.join('1_N/feature/', 'facelib_feature.pkl')
model = lib.face_embedding.FaceModel(args)
dict_face = {}
face_path = '1_N/data/'
face_img = os.listdir(face_path)
index = 0
for img_file in face_img:
    index = index + 1
    print index
    img = cv2.imread(face_path + img_file)
    if np.shape(img) != ():
        f1 = model.get_feature(img)
        dict_face[img_file] = f1
    else:    
        dict_face[img_file] = None
with open(feat_pkl, 'wb') as f:
        cPickle.dump(dict_face, f, cPickle.HIGHEST_PROTOCOL)
