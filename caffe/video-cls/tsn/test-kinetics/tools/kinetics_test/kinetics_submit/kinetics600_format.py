import argparse
import sys
import numpy as np
import json
from util.dict import true_label_file, true_test_file
from collections import Counter

label_dict = true_label_file()
true_test_list = true_test_file()


def json_final(key_result):
    true_result = {"version": "KINETICS-600","results": key_result, "external_data": {
            "used": "true", 
            "details": "Models pre-trained on ImageNet-1k training set", 
        }
        }
    return true_result

def json_style(video_name, video_label, video_score):
    final_result = {video_name: [
              {
              "label": video_label[0],
              "score": video_score[0]
              },
              {
              "label": video_label[1],
              "score": video_score[1]
              },
              {
              "label": video_label[2],
              "score": video_score[2]
              },
              {
              "label": video_label[3],
              "score": video_score[3]
              },
              {
              "label": video_label[4],
              "score": video_score[4]
              }
            ]
            }
    return final_result


result_60000 = np.load(sys.argv[1])
result_12950 = np.load(sys.argv[2])

kinetics_test_file = np.loadtxt('util/new.txt',str,delimiter='\n')
result_72950 = np.append(result_60000['scores'], result_12950['scores'], axis=0)
print np.shape(result_72950)
text_result = open('final_submission_0609.txt','w')

teast_list = []
medium_dict = {}
print len(kinetics_test_file)
for video_index in range(len(kinetics_test_file)):
    if video_index % 1000 == 0:
        print video_index
    if kinetics_test_file[video_index].split(',')[0] + '_' + kinetics_test_file[video_index].split(',')[1] + '_' + kinetics_test_file[video_index].split(',')[2] in true_test_list:
        sort_pre = sorted(enumerate(result_72950[video_index]), key=lambda z:z[1])
        pred_label = [sort_pre[-j][0] for j in range(1,6)]
        pred_score = [round(sort_pre[-j][1], 2) for j in range(1,6)]
        pred_label = [label_dict[str(int(pred_label[q]) + 1)] for q in range(5)]
        text_result.write('"' + kinetics_test_file[video_index].split(',')[0] + '_' + kinetics_test_file[video_index].split(',')[1] + '_' + kinetics_test_file[video_index].split(',')[2] + '": [{"label": "' + pred_label[0] + '", "score": ' + str(pred_score[0]) + '}, {"label": "' + pred_label[1] + '", "score": ' + str(pred_score[1]) + '}, {"label": "' + pred_label[2] + '", "score": ' + str(pred_score[2]) + '}, {"label": "' + pred_label[3] + '", "score": ' + str(pred_score[3]) + '}, {"label": "' + pred_label[4] + '", "score": ' + str(pred_score[4]) + '}], ')
        json_dict = json_style(kinetics_test_file[video_index].split(',')[0] + '_' + kinetics_test_file[video_index].split(',')[1] + '_' + kinetics_test_file[video_index].split(',')[2], pred_label, pred_score)
        medium_dict.update(json_dict)
    else: 
        print kinetics_test_file[video_index].split(',')[0] + '_' + kinetics_test_file[video_index].split(',')[1] + '_' + kinetics_test_file[video_index].split(',')[2]
text_result.close()

#saveJsonFile = 'submission.json'
#with open(saveJsonFile,'w') as write_file:
#    json_result = json_final(medium_dict)
#    write_file.write(json.dumps(json_result, ensure_ascii=False))
#    write_file.flush()

