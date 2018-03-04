### senet（inception-v2/res50/res50-hik/res101/next50/next101/next152）

**1. 模型准确率（Top1/5 err on ImageNet-1k Val)/推理速度及下载地址.**

 Network|224<br/>(single-crop)|F/B(224)|Download<br/>(BaiduCloud)|Source
 :---:|:---:|:---:|:---:|:---:
 se-inception-v2<br/>(se-inception-bn)| 23.64/7.04 | 14.66/10.63ms | [46.0MB](https://pan.baidu.com/s/1qYoPdak) | [senet](https://github.com/hujie-frank/SENet)
 se-resnet50| 22.39/6.37 | 15.29/14.20ms | [107.0MB](https://pan.baidu.com/s/1gf5wsLl) | [senet](https://github.com/hujie-frank/SENet)
 se-resnet50-hik| 21.98/5.80 | 15.37/14.80ms | [107.0MB](https://pan.baidu.com/s/1eSzT6KU) | [senet-hik](https://github.com/shicai/SENet-Caffe)
 se-resnet101| 21.76/5.72 | ../..ms | [189.0MB](https://pan.baidu.com/s/1c1FvCWg) | [senet](https://github.com/hujie-frank/SENet)
 se-resnet152| 21.34/5.54 | ../..ms | [256.0MB](https://pan.baidu.com/s/1dFEnSzR) | [senet](https://github.com/hujie-frank/SENet)
 se-resnext50-32x4d| 20.96/5.53 | ../..ms | [105.0MB](https://pan.baidu.com/s/1dFbEmbv) | [senet](https://github.com/hujie-frank/SENet)
 se-resnext101-32x4d| 19.83/4.95 | ../..ms | [187.0MB](https://pan.baidu.com/s/1qY2wjt6) | [senet](https://github.com/hujie-frank/SENet)
 senet<br/>(se-resnext152-64x4d)| 18.67/4.47 | ../..ms | [441.0MB](https://pan.baidu.com/s/1o7HdfAE) | [senet](https://github.com/hujie-frank/SENet)

**2. 模型预处理参数.**

 Network|mean_value|std
 :---:|:---:|:---:
 senet-official | [104.0, 117.0, 123.0] | [1.0, 1.0, 1.0]
 se-resnet50-hik | [103.94,116.78,123.68] | [58.82, 58.82, 58.82]

**3. 模型分析.**

    - 严格意义上senet不是一类模型，而是一个模型部件（Squeeze-and-Excitation），可以作用在任何其他模型结构中，并且效果优异（几乎不增加时间/显存成本）
    - 训练调参在caffe上基本可以复现结果（官网给出了具体的tricks）
    - 推理阶段是作为预训练模型的不错选择，有切身se-resnet50/se-resnet50-hik，兼顾速度与准确率，极力推荐
