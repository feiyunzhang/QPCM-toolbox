import lib.face_embedding
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='lib/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = lib.face_embedding.FaceModel(args)
img = cv2.imread('1_1/test1.jpg')
f1 = model.get_feature(img)
img = cv2.imread('1_1/test2.jpg')
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
