'''
parse predictions from allennlp.article_classifier
python predict.py predictions.txt result.jsonl
'''

import os
import sys
import json
from tqdm import tqdm

LABELS = ['地名', '机构名称', '人名']

def parse(input_data, prediction):
    try:
        in_json = json.loads(input_data.replace('input:', '').strip().replace("'", '"'))
        pred_json = json.loads(prediction.replace('prediction:', '').strip().replace("'", '"'))
    except:
        return None

    return {
        'word': in_json['title'],
        'label': in_json['label'],
        'pred': pred_json['label'],
        'score': max(pred_json['class_probabilities'])
    }

def process(pred_file):
    with open(pred_file) as fp:
        l_input = ''
        l_pred  = ''
        for line in tqdm(fp):
            if line.startswith('input:'):
                l_input = line
                l_pred  = ''
            elif line.startswith('prediction:'):
                l_pred  = line
            if l_input != '' and l_pred != '':
                yield parse(l_input, l_pred)
                l_input = ''
                l_pred  = ''


def save_jsonl(data, output):
    with open(output, 'w') as fp:
        for item in data:
            if item is None or item['pred'] not in LABELS:
                continue
            if item['label'] == item['pred'] and item['score'] > 0.7:
                item.pop('pred')
                fp.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    save_jsonl(process(sys.argv[1]), sys.argv[2])
