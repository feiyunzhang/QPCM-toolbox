python main.py kinetics Flow kinetics600-flow-datafile/kinetics_600_train_flow.txt kinetics600-flow-datafile/kinetics_600_val_6000_flow.txt  \
  --arch InceptionV3 --num_segments 3  --gd 20 --lr 0.01 --lr_steps 70 190 300 --epochs 340 \
  -b 144 -j 4 --dropout 0.7 --snapshot_pref kinetics_inceptionv3_flow_ --flow_pref flow_