# TSN

Temporal Segment Networks用于视频动作分类。

### Install

从`action_recog`的caffe镜像（`reg-xs.qiniu.io/atlab/base/caffe/gpu.action:example`）创建一个容器，进入到容器中。运行：

```
git clone http://github.com/qbox/Alg-VideoAlgorithm
cd Alg-VideoAlgorithm/video-classification/temporal-segment-networks
```

### Data Preparation

1. 获取视频数据

   目前支持的数据集包括UCF101，HMDB51，Activitynet1.2/1.3。其中，UCF101和Activity1.3的视频数据已经存储在qiniu云存储上了，具体位置可以在[视频数据集整理](https://cf.qiniu.io/pages/viewpage.action?pageId=62049218)的cf页面查询。

2. 获取数据集标注信息

   数据集的标注信息可以直接从对应数据集上下载，我存了一份到qiniu云存储，包括UCF101，HMDB51，Activitynet1.2/1.3三个数据集的类别信息，标注信息，数据集划分（splits）等信息，运行以下脚本下载：

   ```
   bash scripts/get_dataset.sh
   ```

3. 截帧 & 提取密集光流

   数据预处理的工作包括截帧和提取密集光流，处理代码在`basical-tools/dense-flow`中，需要用的可以到对应目录。实际上，我已经提取好了UCF101的帧数据，Activity1.3的帧和密集光流数据也在提取中，可以直接使用，存储位置可以在[视频数据集整理](https://cf.qiniu.io/pages/viewpage.action?pageId=62049218)的cf页面查询。

### Training

以UCF101在BN-Inception上训练Spatial Network为例。

1. 获取pretrained model：

   ```
   bash scripts/get_init_models.sh
   ```

2. 生成训练集和测试集的file lists：

   由于训练时的输入数据依赖于caffe的`VideoDataLayer`层，这个层需要指定一个file list作为其数据来源。file list的每一行包含每个视频的帧存储位置，视频帧数，视频的groudtruth类别。例如，一个file list长这样：

   ```
   /workspace/data/UCF-frames/v_HorseRace_g11_c02 279 40
   /workspace/data/UCF-frames/v_Rowing_g10_c01 481 75
   /workspace/data/UCF-frames/v_PlayingTabla_g12_c03 256 65
   /workspace/data/UCF-frames/v_BandMarching_g21_c01 311 5
   ...
   ```

   要构建file list，运行以下脚本：

   ```
   bash scripts/build_file_list.sh ucf101 FRAME_PATH
   ```

   生成的file list存储在`data/`目录下，命名规则如`ucf101_rgb_train_split_1.txt`。

3. 开始训练：

   ```
   bash scripts/train_tsn.sh ucf101 rgb GPU_NUM
   ```

   训练产生的模型存储在`models/`目录下。

### Testing

to be continued ...



### Reference

[1] https://github.com/yjxiong/temporal-segment-networks