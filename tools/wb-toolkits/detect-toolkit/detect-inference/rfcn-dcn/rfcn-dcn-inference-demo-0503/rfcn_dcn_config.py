rfcn_dcn_config = {
    "config_yaml_file": "/workspace/data/BK/rfcn/Deformable-ConvNets/demo/models/resnet_v1_101_terror_dcn_rfcn_end2end_ohem.yaml",
    "modelParam": {
        "modelBasePath": "/workspace/data/BK/rfcn/Deformable-ConvNets/demo/models",
        "epoch":14
    },
    "one_batch_size":200,
    'num_classes':11,
    'num_classes_name_list': ['__background__',
                              'islamic flag', 'isis flag', 'tibetan flag', 'knives_true', 'guns_true',
                              'knives_false', 'knives_kitchen',
                              'guns_anime', 'guns_tools',
                              'not terror'],
    'need_label_dict': {
        1: 'islamic flag',
        2: 'isis flag',
        3: 'tibetan flag',
        4: 'knives_true',
        5: 'guns_true',
        6: 'knives_false',
        7: 'knives_kitchen',
        8: 'guns_anime',
        9: 'guns_tools',
        10: 'not terror'
    },
    'need_label_thresholds': {
        1: 0.7,
        2: 0.7,
        3: 0.7,
        4: 0.6,
        5: 0.6,
        6: 0.6,
        7: 0.5,
        8: 0.5,
        9: 0.5,
        10: 1.0
    }
}
