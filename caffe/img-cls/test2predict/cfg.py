class Config:
    PLATFORM = "GPU"
    TEST_GPU_ID = 0
    CLS_NET_DEF_FILE = 'models/deploy.prototxt'
    CLS_MODEL_PATH = 'models/XXX.caffemodel'
    CLS_CONFIDENCE_THRESH = 0.8
    CLS_LABEL_INDEX = 'models/label_index.txt'
