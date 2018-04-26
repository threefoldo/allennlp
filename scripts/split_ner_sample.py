'''
convert NER sample from word/tag to ch/tag.
'''

import os
import sys
import json
import random

def conv(in_file):
    with open(in_file) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            output = []
            for item in line.split():
                word, tag = item.split('/')
                output.extend(['%s/%s' % (ch, tag) for ch in word])
            yield output

def process(in_file):
    data = list(conv(in_file))
    for i in range(5):
        random.shuffle(data)

    test_num = int(len(data) / 10)
    return data[:-test_num], data[-test_num:]

def save(data, out_file):
    with open(out_file, 'w') as fp:
        for item in data:
            fp.write(' '.join(item) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python conv_ner_sample.py data/ner_sample1.txt')
        sys.exit(0)
    in_file = sys.argv[1]
    basename = in_file.replace('.txt', '')
    train, dev = process(in_file)
    save(train, basename + '_train.txt')
    save(dev,   basename + '_dev.txt')


