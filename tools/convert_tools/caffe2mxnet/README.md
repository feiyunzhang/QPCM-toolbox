# caffe模型转换mxnet工具

安装编译:

    - make 
    - 设置py脚本中环境变量

用法：

    - python convert_model.py caffe_prototxt caffe_caffemodel save_model_name

示例：

    - python convert_model.py demo/SE-ResNeXt-101.prototxt demo/SE-ResNeXt-101.caffemodel SE-ResNeXt-101