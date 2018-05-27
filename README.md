# Andrew-toolbox

Andrew's toolbox, a collection of frequently used scripts.

## Content

List of available tools 
1. `mxnet/` - MXNet tools
      * `mxnet-cam` - Draw class activation mapping
      * `mxnet-visualizer` - Draw curves according to training log
      * `img-cls` - Image-classification task
      * `obj-det` - Object-detection task
      * `recordio_traverse.py` - Traverse a RecordIO file 
      * `mxnet_setup.sh` - Auto-install mxnet
2. `caffe/` - Caffe tools
      * `caffe-visualizer` - Draw curves according to training log
      * `caffe-fm-visualizer` - Visualize internal featuremap
      * `img-cls` - Image-classification task
3. `pytorch/` - Pytorch tools
4. `labelX/` - LabelX tools
      * `gen_labelx_jsonlist.py` - Generate labelX-standard jsonlist
      * `labelx_jsonlist_adapter.py` - Convert labelX-standard jsonlist to useable format
5. `modelzoo/` - Model Zoo
6. `evaluation` - Evaluate image classification/detection results
7. `tools/` - Some useful toolkits

## Requirements

* Most scripts require docopt interface pack:

    ```
    pip install docopt
    ```

* If `No module named AvaLib` is warned:

    ```
    # either remove this line and corresponding lines in codes
    import AvaLib

    # or install AvaLib with cmd
    easy_install lib/AvaLib-1.0-py2.7.egg
    ```

## Usage

use `xxx.py -h` to get help information
