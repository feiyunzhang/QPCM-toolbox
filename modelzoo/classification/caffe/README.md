# 基于caffe框架的图像分类预训练模型
1. 所有模型均是通过caffe-inference-all/框架完成推理测试
2. 每个模型Forward/Backward推理配置：
    - mini-batch=1
    - cuda8.0
    - cuDNN 5.1（与cuDNN 6.0速度相当）
    - Pascal Titan X GPU(与P40/P100相近，比K80快30%)
    - 迭代1000次取平均
3. 推理样例代码：
    - python evaluation_cls.py
4. 说明：
    - 目前模型链接都在baidu网盘，后面会移到bucket上
    - 后面会陆续补充更新更多的模型，为大家训练模型提供选择

