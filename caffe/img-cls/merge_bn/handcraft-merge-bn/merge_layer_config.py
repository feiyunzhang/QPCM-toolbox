merge_conv_layer_dict = {
    # key : type string 
    #       conv layer name
    # element : type list
    #       [batchnorm layer name,scala layer name]
    # refindet res18 cls mix det model
    'conv1': ['bn_conv1', 'scale_conv1'],
    'res2a_branch2a': ['bn2a_branch2a', 'scale2a_branch2a'],
    'res2a_branch2b': ['bn2a_branch2b', 'scale2a_branch2b'],
    'res2b_branch2a': ['bn2b_branch2a', 'scale2b_branch2a'],
    'res2b_branch2b': ['bn2b_branch2b', 'scale2b_branch2b'],
    'res3a_branch1': ['bn3a_branch1', 'scale3a_branch1'],
    'res3a_branch2a': ['bn3a_branch2a', 'scale3a_branch2a'],
    'res3a_branch2b': ['bn3a_branch2b', 'scale3a_branch2b'],
    'res3b_branch2a': ['bn3b_branch2a', 'scale3b_branch2a'],
    'res3b_branch2b': ['bn3b_branch2b', 'scale3b_branch2b'],
    'res4a_branch1': ['bn4a_branch1', 'scale4a_branch1'],
    'res4a_branch2a': ['bn4a_branch2a', 'scale4a_branch2a'],
    'res4a_branch2b': ['bn4a_branch2b', 'scale4a_branch2b'],
    'res4b_branch2a': ['bn4b_branch2a', 'scale4b_branch2a'],
    'res4b_branch2b': ['bn4b_branch2b', 'scale4b_branch2b'],
    'res5a_branch1': ['bn5a_branch1', 'scale5a_branch1'],
    'res5a_branch2a': ['bn5a_branch2a', 'scale5a_branch2a'],
    'res5a_branch2b': ['bn5a_branch2b', 'scale5a_branch2b'],
    'res5b_branch2a': ['bn5b_branch2a', 'scale5b_branch2a'],
    'res5b_branch2b': ['bn5b_branch2b', 'scale5b_branch2b'],
    'res4a_branch1_cls': ['bn4a_branch1_cls', 'scale4a_branch1_cls'],
    'res4a_branch2a_cls': ['bn4a_branch2a_cls', 'scale4a_branch2a_cls'],
    'res4a_branch2b_cls': ['bn4a_branch2b_cls', 'scale4a_branch2b_cls'],
    'res4b_branch2a_cls': ['bn4b_branch2a_cls', 'scale4b_branch2a_cls'],
    'res4b_branch2b_cls': ['bn4b_branch2b_cls', 'scale4b_branch2b_cls'],
    'res5a_branch1_cls': ['bn5a_branch1_cls','scale5a_branch1_cls'],
    'res5a_branch2a_cls': ['bn5a_branch2a_cls', 'scale5a_branch2a_cls'],
    'res5a_branch2b_cls': ['bn5a_branch2b_cls', 'scale5a_branch2b_cls'],
    'res5b_branch2a_cls': ['bn5b_branch2a_cls', 'scale5b_branch2a_cls'],
    'res5b_branch2b_cls': ['bn5b_branch2b_cls', 'scale5b_branch2b_cls']
}
