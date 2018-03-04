### Inception（V3/V4/I-R-V2/X）

**1. 模型准确率（Top1/5 err on ImageNet-1k Val)/推理速度及下载地址.**

 Network|299<br/>(single-crop)|F/B(299)|Download<br/>(BaiduCloud)|Source
 :---:|:---:|:---:|:---:|:---:
 inception-v3| 21.67/5.75 | 21.79/19.82ms | [91.1MB](https://pan.baidu.com/s/1boC0HEf) | [mxnet](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md)
 xception| 20.90/5.49 | 14.03/30.39ms | [87.4MB](https://pan.baidu.com/s/1gfiTShd) | [keras-models](https://github.com/fchollet/deep-learning-models)
 inception-v4| 20.03/5.09 | 32.96/32.19ms | [163.1MB](https://pan.baidu.com/s/1c6D150) | [tf-slim](https://github.com/tensorflow/models/tree/master/slim)
 inception-resnet-v2| 19.86/4.83 | 49.06/54.83ms | [213.4MB](https://pan.baidu.com/s/1jHPJCX4) | [tf-slim](https://github.com/tensorflow/models/tree/master/slim)

**2. 模型预处理参数.**

 Network|mean_value|std
 :---:|:---:|:---:
 inception-all | [128.0, 128.0, 128.0] | [128.0, 128.0, 128.0] 

**3. 模型分析.**

    - 是谷歌先后同步于vgg/resnet设计的网络架构，结构设计有海量机器狂调参的影子
    - 先后出过5个（不算xception）版本，论文给出的trick不全（也有故意错误）
    - 训练较难复现，尤其是inception v4与inception-resnet-v2这2个模型，基本复现不了，都是从tf转过来的
    - 与resnet差异性较大，是模型融合的不错选择
    - 用于pretrain model较大（相比resnet），不建议使用
