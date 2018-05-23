config={
    "model":{
        "deploy_file":
        "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/mobilenetv2_mergebn_deploy_batch.prototxt",
        "weight_file":
        "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/mobilenetv2-v0.3-t3-mergebn.caffemodel",
        # csv file ,sep is ,
        "label_file": "/workspace/data/opt-model-speed-Dir/cls/mobilenet-v2-caffe/models/labels.lst",
        "batch_size":16
    },
    "dataParam":{
        "resize":256,
        "input_size":224,
        "mean_data_list": [103.94, 116.78, 123.68],
        "scala": 0.017
    }
}

