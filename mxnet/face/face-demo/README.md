
# face demo
### Requirements
      
   1. Install `MXNet` with GPU support(Python 2.7). 

       ```
       pip install mxnet-cu80 or pip install mxnet-cu90
       ```
   2. Install 'Caffe' with GPU sipport(Python 2.7).

       ```
       参照：http://caffe.berkeleyvision.org/install_apt.html
       ```
   3. Inatall the third software.

       ```
       pip install easydict
       ```
      

### 用法：

   1. 人脸检测 

       ```
       python det-demo.py
       ```
   2. 人脸识别 1：1

       ```
       python 1_1-demo.py
       ```  
   3. 人脸识别 1：N

      提取底库数据特征
       ```
       python 1_N_facelib.py
       ```  

      比对测试数据与底库特征
       ```
       python 1_N_test.py
       ```  