# v1.0 Cascade R-CNN

## TODO
-[x] 初始化
-[] 边框均值方差
-[x] 各阶段权重
-[x] 4 gpus

## FAQ
F: rcnn 子网是类别无关的吗 ?
A: 是类别无关的. 在 cascade-rcnn-caffe 的 ProposalTargetLayer 层的 cpp 源代码中, bbox_cls_aware == False

F: 各阶段边框均值方差是多少 ?
A:
```
stage 2:
bbox_reg_param {
    bbox_mean: 0 bbox_mean: 0 bbox_mean: 0 bbox_mean: 0
    bbox_std: 0.1 bbox_std: 0.1 bbox_std: 0.2 bbox_std: 0.2
  }
stage 3:
bbox_reg_param {
    bbox_mean: 0 bbox_mean: 0 bbox_mean: 0 bbox_mean: 0
    bbox_std: 0.05 bbox_std: 0.05 bbox_std: 0.1 bbox_std: 0.1
  }
```

F: fg_fraction 是多少 ?
A: 0.25

F: 前背景 roi 阈值是多少 ?
A:
```
stage 1:
    fg_thr: 0.5
    bg_thr_hg: 0.5
    bg_thr_lw: 0.0
stage 2:
    fg_thr: 0.6
    bg_thr_hg: 0.6
    bg_thr_lw: 0.0
stage 3:
    fg_thr: 0.7
    bg_thr_hg: 0.7
    bg_thr_lw: 0.0
```

F: 各阶段 loss 权重 ?
A:
```
1*rpn_loss
+ 1*loss_cls + 1*loss_box
+ 0.5*los_cls_2nd + 0.5*loss_box_2nd
+ 0.25*los_cls_3rd + 0.25*loss_box_3rd

```

F: mxnet 中, 在何处计算 loss ?
A: 在 resnet_v1_101_cascade_rcnn.py 的 Group 中

F: Deformable-ConvNets 中, 怎么使用边框均值和方差 ?
``` python
-- faster_rcnn.py:
from core import callback, metric
means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback, ...)

-- callback.py
def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('bbox_pred_weight_test')
        arg.pop('bbox_pred_bias_test')
    return _callback
```

## 代码改动
复制 resnet_v1_101_rcnn.py 并重命名为 faster_rcnn/symbols/resnet_v1_101_cascade_rcnn.py, 并做如下改动:
修改 get_symbol 函数, 以搭建级联网络结构.

修改 rcnn.py:
新建函数 sample_rois_stage2 与 sample_rois_stage3, 为不同阶段设定不同的前背景 roi 阈值.

修改 proposal_target.py:
修改 forward 函数, 使得不同阶段能够运行不同的 sample_rois 函数.

创建新层 faster_rcnn/operator_py/decode_bbox.py, 用于将上一阶段的 bbox_pred 转为下一阶段的 proposals(绝对坐标)
传入参数: 上层的 rois, bbox_pred, 图像尺寸
