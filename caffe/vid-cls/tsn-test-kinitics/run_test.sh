python tools/eval_net.py kinetics400 1 rgb data/kinetics400_splits/data \
    models/tsn_se_resnext_101_32x4d_rgb_deploy.prototxt models/kinetics_se_resnext101_32x4d_t3_iter_85000.caffemodel \
    --num_worker 2 --save_scores score_file/se_next101_t3_rgb_340x256_mini --caffe_path lib/caffe
