python main_nvvl.py kinetics RGB videodatafile-0424/train_new_0318_255737_withlabel.txt  videodatafile-0424/val_new_0318_4000_withlabel.txt --arch dpn107 --num_segments 3 -p 20 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 96 -j 2 --dropout 0.8 --snapshot_pref kinetics_dpn107_