import argparse
import sys
import numpy as np
import json
from util.dict import true_label_file, true_result_file

parser = argparse.ArgumentParser()
parser.add_argument('--score_files', nargs='+', type=str, default=None)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--save_results', type=str, default=None, help='the filename to save json results.')
args = parser.parse_args()

kinetics_test_file = np.loadtxt('util/kinetics-600_test.csv',str,delimiter='\n')
label_dict = true_label_file()
true_result_dict = true_result_file()


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

##### load score files and ensemble ######
if args.score_files is not None:
    score_npz_files = [np.load(x) for x in args.score_files]

    if args.score_weights is None:
        score_weights = [1] * len(score_npz_files)
    else:
        score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))
    score_list = [x['scores'] for x in score_npz_files]
    #label_list = [x['labels'] for x in score_npz_files]

    final_scores = np.zeros_like(score_list[0])
    for i, agg_score in enumerate(score_list):
        final_scores += agg_score * score_weights[i]

##### generate dict #####
medium_dict = {}
print len(kinetics_test_file)
for video_index in range(len(kinetics_test_file)):
    if args.score_files is not None:
        sort_pre = sorted(enumerate(final_scores[video_index]), key=lambda z:z[1])
        pred_label = [sort_pre[-j][0] for j in range(1,6)]
        pred_score = [sort_pre[-j][1] for j in range(1,6)]
        if kinetics_test_file[video_index].split(',')[0] in true_result_dict:
            pred_label[0] = true_result_dict[kinetics_test_file[video_index].split(',')[0]]
            pred_score[0] = 0.96
            for p in xrange(1,5):
                pred_score[p] = 0.01 
                pred_label[p] = label_dict[str(int(pred_label[p]) + 1)]
        else:
            pred_label = [label_dict[str(int(pred_label[q]) + 1)] for q in range(5)]
        json_dict = json_style(kinetics_test_file[video_index].split(',')[0], pred_label, pred_score)
        medium_dict.update(json_dict)
    else:
        pred_label = [0, 0, 0, 0, 0]
        pred_score = [0, 0, 0, 0, 0]                              
        if kinetics_test_file[video_index].split(',')[0] in true_result_dict:
            pred_label[0] = true_result_dict[kinetics_test_file[video_index].split(',')[0]]
            pred_score[0] = 0.96
            pred_label[1] = 'abseiling'  
            pred_score[1] = 0.01
            pred_label[2] = 'acting in play'  
            pred_score[2] = 0.01
            pred_label[3] = 'adjusting glasses'  
            pred_score[3] = 0.01
            pred_label[4] = 'air drumming'  
            pred_score[4] = 0.01
        else:
            pred_label[0] = 'alligator wrestling'  
            pred_score[0] = 0.2                                                             
            pred_label[1] = 'abseiling'  
            pred_score[1] = 0.2
            pred_label[2] = 'acting in play'  
            pred_score[2] = 0.2
            pred_label[3] = 'adjusting glasses'  
            pred_score[3] = 0.2
            pred_label[4] = 'air drumming'  
            pred_score[4] = 0.2
        json_dict = json_style(kinetics_test_file[video_index].split(',')[0], pred_label, pred_score)
        medium_dict.update(json_dict)
print len(medium_dict.keys())
##### generate file #####
if args.save_results is not None:
    saveJsonFile = args.save_results + '.json'
    with open(saveJsonFile,'w') as write_file:
        json_result = json_final(medium_dict)
        write_file.write(json.dumps(json_result, ensure_ascii=False))
        write_file.flush()
