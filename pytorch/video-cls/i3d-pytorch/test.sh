python test_models.py kinetics RGB datafile/val_new_0318_4000.txt \
                      kinetics_i3dresnet101__rgb_model_best.pth.tar  \
                      --arch i3d_resnet101 \
                      --test_clips 10   \
                      --save_score i3d-120e-model-best-score-10crop  \
                      --gpus 1 -j 1