#batch-size=1
#first modify the prototxt/caffemodel and mean/std
python caffe_image_classify.py --img_file test.lst --root val/ --gpu 0


#batch-size>1

python caffe_image_classify-multibatch.py test.lst out.log --arch deploy.prototxt --weights XXXX.caffemodel --label labels.lst --model-root final-model/ --data-prefix val/ --img-width 224 --gpu 0
