### Resnext（26/50/101）

**1. 模型准确率（Top1/5 err on ImageNet-1k Val)/推理速度及下载地址.**

 Network|224<br/>(single-crop)|F/B(224)|Download<br/>(BaiduCloud)|Source
 :---:|:---:|:---:|:---:|:---:
 resnext26-32x4d| 24.93/7.75 | 8.53/10.12ms | [58.9MB](https://pan.baidu.com/s/1dFzmUOh) | [pytorch-cls](https://github.com/soeaver/pytorch-classification)
 resnext50-32x4d| 22.37/6.31 | 17.29/20.08ms | [95.8MB](https://pan.baidu.com/s/1kVqgfJL) | [facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101-32x4d| 21.30/5.79 | 30.73/35.75ms | [169.1MB](https://pan.baidu.com/s/1hswrNUG) | [facebookresearch](https://github.com/facebookresearch/ResNeXt)
 resnext101-64x4d| 20.60/5.41 | 42.07/64.58ms | [319.2MB](https://pan.baidu.com/s/1pLhk0Zp) | [facebookresearch](https://github.com/facebookresearch/ResNeXt)

**2. 模型预处理参数.**

 Network|mean_value|std
 :---:|:---:|:---:
 resnext-all | [103.52, 116.28, 123.675] | [57.375, 57.12, 58.395]
 
**3. 模型分析.**

    - 是resnet的升级版本，同等规模下比resnet效果更好
    - Cardinality越大，模型规模越大，效果越好，见resnext101-32x4d与resnext101-64x4d
    - 训练调参在pytorch上较容易，在caffe上比较难复现结果，上述模型均是从pytorch转换过来的
    - 用于pretrain model较大（同样层数下相比resnet），慎重使用
