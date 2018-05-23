python test_models.py kinetics RGB datafile/val_new_0318_4000.txt \
                      nasnet_80e.pth.tar \
                      --arch nasnetalarge \
                      --save_score nasnetlarge__12_score.txt  \
                      --gpus 0 -j 1