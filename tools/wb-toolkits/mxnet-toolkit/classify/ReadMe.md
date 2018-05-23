# Train
## preProcess data
```
python /opt/mxnet/tools/im2rec.py --list True --recursive True ./data/Terror-Classify-V0.3-train ./data/train
python /opt/mxnet/tools/im2rec.py --list True --recursive True ./data/Terror-Classify-V0.3-val ./data/val
参数说明：
./data/Terror-Classify-V0.3-val : the result lst file
./data/val ：指向数据存放的目录，val 这个目录下存放类别子目录。分类类别是按照 类别文件名排序获得。

```
```
python /opt/mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 ./data/Terror-Classify-V0.3-train.lst ./data/train
python /opt/mxnet/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 ./data/Terror-Classify-V0.3-val.lst ./data/val
```

