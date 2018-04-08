python main.py kinetics RGB kinetics_files/train_rgb_final.txt  kinetics_files/val_rgb_final.txt --arch resnet101 --snapshot_pref kinetics_i3dresnet101_
                --lr 0.001 --lr_steps 45 90 --epochs 120
                -b 128 -j 8 --dropout 0.5 -p 20 --gd 20