# -*- coding:utf-8 -*-
"""
    这个脚本的作用 :
    labelx format --- regression format 相互转换
    labelx format :
        {
            "url": "http://p6n248jl3.bkt.clouddn.com/detv09-and-flag03037/jpg/n02749479_7473.jpg", 
            "type": "image", 
            "label": [
                {
                    "name": "detect", 
                    "type": "detection", 
                    "version": "1", 
                    "data": [
                        {"class": "guns_true", "bbox": [[54, 101], [154, 101], [154, 126], [54, 126]], "ground_truth": true}, 
                        {"class": "guns_true", "bbox": [[58, 62], [115, 62], [115, 82], [58, 82]], "ground_truth": true}, 
                        {"class": "guns_true", "bbox": [[64, 55], [113, 55], [113, 78], [64, 78]], "ground_truth": true}, 
                        {"class": "guns_true", "bbox": [[57, 101], [154, 101], [154, 126], [57, 126]], "ground_truth": true}
                    ]
                }
            ]
        }
    regression format :
        weiboimg-2017-11-17-10-28-aHR0cHM6Ly93dzEuc2luYWltZy5jbi9vcmozNjAvM2ZmNWVjYjdqdzFmMW05NXYzdHh2ajIwcW8wdzJ0aWQuanBn.jpg\t[{"index": 1, "score": 0.9603773355484009, "pts": [[0, 33], [318, 33], [318, 418], [0, 418]], "class": "person"}]
"""
import json
def get_jsonList_line_labelInfo(line=None):
    """
    用于获取打标过的一行jsonlist 包含的 label 信息
    detect
    value is list : 
        the element in the list is : 
            bbox list : 
                [
                    {"class":"knives_true","bbox":[[43,105],[138,105],[138,269],[43,269]],"ground_truth":true},
                    {"class":"guns_true","bbox":[[62,33],[282,33],[282,450],[62,450]],"ground_truth":true},
                    {"class":"guns_true","bbox":[[210,5],[399,5],[399,487],[210,487]],"ground_truth":true}
                ]
    return key:value 
            key is url
            value : label info ()
    """
    key = None
    value = None
    line_dict = json.loads(line)
    key = line_dict['url']
    if line_dict['label'] == None or len(line_dict['label']) == 0:
        return key, None
    label_dict = line_dict['label'][0]
    if label_dict['data'] == None or len(label_dict['data']) == 0:
        return key, None
    data_dict_list = label_dict['data']
    label_bbox_list_elementDict = []
    for bbox in data_dict_list:
        if 'class' not in bbox or bbox['class'] == None or len(bbox['class']) == 0:
            continue
        label_bbox_list_elementDict.append(bbox)
    if len(label_bbox_list_elementDict) == 0:
        value = None
    else:
        value = label_bbox_list_elementDict
    return key, value


def labelxFormat_2_regressionFormat(labelx_line=None):
    """
        
        regression format : bbox info not include index and score
    """
    if labelx_line is None:
        return (False, None)
    image_name, bbox_list = get_jsonList_line_labelInfo(line=labelx_line)
    if bbox_list is None:
        rg_bboxs_list = []
    else:
        rg_bboxs_list = []
        for i_bbox in bbox_list:
            rg_bbox = dict()
            rg_bbox['class'] = i_bbox['class']
            rg_bbox['pts'] = i_bbox['bbox']
            rg_bboxs_list.append(rg_bbox)
        rg_bboxs_list = []
    regressionLine = "%s\t%s" % (image_name, json.dumps(rg_bboxs_list))
    return regressionLine


def createLabelxFormatDict(url=None, bboxDataList=None):
    """
        url : image url
        bboxDataList : list , 
                        element is : dict
                        element_dict : {
                                            "class": "label class name",
                                            "ground_truth": true,
                                            "bbox": [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
                                       }

        return : labelx_dict
    """
    labelx_dict = dict()
    labelx_dict['url'] = url
    labelx_dict['type'] = "image"
    one_dict = dict()
    one_dict['name'] = 'detect'
    one_dict['type'] = 'detection'
    one_dict['version'] = '1'
    one_dict['data'] = bboxDataList
    labelx_dict_label_list = []
    labelx_dict_label_list.append(one_dict)
    labelx_dict['label'] = labelx_dict_label_list
    return labelx_dict


def regressionFormat_2_labelxFormat(regressionLine=None):
    """
        input is : regression format line
        output is : labelx format line
    """
    imageName_or_url = regressionLine.split('\t')[0]
    rg_bboxs_line = regressionLine.split('\t')[-1]
    rg_bboxs_list = json.loads(rg_bboxs_line)
    labelx_bboxs_list = []
    for rg_i_bbox in rg_bboxs_list:
        labelx_bbox = dict()
        labelx_bbox['class'] = rg_i_bbox['class']
        labelx_bbox['bbox'] = rg_i_bbox['pts']
        labelx_bbox['ground_truth'] = True
        labelx_bboxs_list.append(labelx_bbox)
    labelx_dict = createLabelxFormatDict(url=imageName_or_url,
                                         bboxDataList=labelx_bboxs_list)
    return json.dumps(labelx_dict)
