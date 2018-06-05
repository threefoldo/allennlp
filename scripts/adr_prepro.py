import json
import re
import os
import sys
import random
import spacy
from spacy.tokenizer import Tokenizer
from pipeline import Pipeline, FileReader, ConllWriter, LineOps, ListOps, DictOps

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)
    

def parse_2cols(line):
    '''
    two columns: tweet id, text
    '''
    items = line.strip().split('\t')
    if len(items) < 2:
        # print('invalid tweet:', items)
        return None
    return {'id': items[0], 'text': ' '.join(items[1:])}

def parse_4cols(line):
    '''
    four columns: tweet id, user id, ann id, text
    '''
    items = line.strip().split('\t')
    if len(items) < 4 or items[3].find('not exist') >= 0:
        # print('invalid ann tweet: ', items)
        return None
    return {'tid': items[0], 'uid': items[1], 'ann_id': items[2], 'text': items[3]}

def parse_annotation(line):
    '''
    7 cols: ann, start, end, type, reaction, drug1, drug2
    '''
    items = line.strip().split('\t')
    if len(items) < 7:
        # print('invalid annotation: ', items)
        return None
    return {
        'ann_id': items[0],
        'start': items[1],
        'end': items[2],
        'type': items[3],
        'reaction': items[4],
        'drug1': items[5],
        'drug2': items[6],
    }

def filter_invalid(batch):
    return [item for item in batch if item]

def read_tweets(filename):
    reader = FileReader(filename, batch_size=100)
    pipe = Pipeline(reader, [parse_2cols])
    lines = pipe.run()
    print(len(lines), lines[0])
    return lines

def read_ann_tweets(filename):
    reader = FileReader(filename, batch_size=100)
    pipe = Pipeline(reader, [parse_4cols])
    lines = pipe.run()
    print(len(lines), lines[0])
    return lines

def read_annotations(filename):
    reader = FileReader(filename, batch_size=100)
    pipe = Pipeline(reader, [parse_annotation], [filter_invalid])
    lines = pipe.run()
    print(len(lines), lines[0])
    return lines

def align_tweets_annotations(tweets, annotations):
    # build tweets map
    tweets_map = {}
    for t in tweets:
        ann_id = t['ann_id']
        tweets_map[ann_id] = t
    
    # find the tweet for each annotation
    # one tweet may have multiple annotations, each annotation as a training sample
    # add drug name to the tweet
    result = []
    for ann in annotations:
        ann_id = ann['ann_id']
        tweet = tweets_map.get(ann_id)
        if tweet is None or tweet['text'].startswith('not exist'):
            # print('ann %s is invalid' % ann_id)
            continue
        new_ann = {
            'ann_id': ann['ann_id'],
            'type': ann['type'],
            'reaction': ann['reaction'],
            'drug1': ann['drug1'],
            'drug2': ann['drug2'],
            'text': tweet['text']
        }
        result.append(new_ann)
    return result


re_ellipse = re.compile('\.\.\.+')

def check_ellipse(token):
    '''
    tokenizer cannot split ... from words
    '''
    token = re_ellipse.sub('...', token)
    new_tokens = []
    for item in token.split('...'):
        if len(item) > 0:
            new_tokens.append(item.strip())
            new_tokens.append('...')

    # if token is '...'
    if re_ellipse.match(token) and len(new_tokens) < 1:
        return ['...']

    if token.endswith('...'):
        return new_tokens
    return new_tokens[:-1]

def split_mark(token):
    '''
    check punctation marks
    '''
    # check special cases
    if token == '...':
        return [token]

    new_tokens = []
    last_token = ''
    for ch in token:
        if ch in [',', '!', ':', '~', '"', '/', '?', '(', ')']:
            if len(last_token) > 0:
                new_tokens.append(last_token)
            if ch != '/':
                new_tokens.append(ch)
            last_token = ''
        else:
            last_token += ch

    # check period mark '.' at the end
    if len(last_token) > 0:
        if last_token[-1] == '.':
            new_tokens.append(last_token[:-1])
            new_tokens.append('.')
        else:
            new_tokens.append(last_token)
    return new_tokens

def normalize_patterns(token):
    if token.startswith('@'):
        return ['@name']
    if token.startswith('#'):
        return [token[1:]]
    # token = token.replace('/', '*')
    token = token.replace('&#39;', "'").replace('&amp;', '>')
    return [token]


def transform(text):
    '''
    pass through all transform which accept a string and return a list
    '''
    tokens, new_tokens = [text], []
    for tr in [check_ellipse, split_mark, normalize_patterns]:
        for tok in tokens:
            if tok and len(tok) > 0:
                new_tokens.extend(tr(tok))
        tokens, new_tokens = new_tokens, []
    return tokens


def tag_span(text, tag):
    '''
    mark all tokens in text with a specific tag
    '''
    tokens = []
    for tok in tokenizer(text):
        tokens.append(tok.text)

    if tag != 'O':
        return [(tok, 'I-' + tag) for tok in tokens]
    return [(tok, 'O') for tok in tokens]    

def mark_annotations(annotations):
    '''
    transform text and mark spans
    '''
    result = []
    for ann in annotations:
        reaction = ann['reaction'].lower()
        text = ann['text'].lower()
        
        # transform text
        tokens = []
        for tok in tokenizer(text):
            tokens.extend(transform(tok.text))
        text = ' '.join(tokens)
        pos = text.find(reaction)
        if pos < 1:
            continue
        spans = tag_span(text[:pos], 'O')
        spans.extend(tag_span(text[pos : pos +len(reaction)], 'ADR'))
        spans.extend(tag_span(text[pos +len(reaction):], 'O'))
        result.append(spans)
    return result

def show_marked(marked):
    print(' '.join(['%s/%s' % (w[0], w[1]) for w in marked]))
    
def fold5_split(train, test, basedir):
    all_data = train
    all_data.extend(test)
    total = len(all_data)
    chunk = int(total / 5)
    random.shuffle(all_data)
    
    for i in range(5):
        start, end = i * chunk, (i + 1) * chunk
        print('split %d : %d, %d' % (i + 1, start, end))
        train_data = []
        test_data = []
        for k in range(len(all_data)):
            if k >= start and k < end:
                test_data.append(all_data[k])
            else:
                train_data.append(all_data[k])
        writer = ConllWriter('%s%d/train.txt' % (basedir, i + 1))
        writer(train_data)
        writer = ConllWriter('%s%d/test.txt' % (basedir, i + 1))
        writer(test_data)

def read_ner0531():
    tweets_all = read_ann_tweets('ner0531/data_all.txt')
    ann_test = read_annotations('ner0531/test_tweet_annotations.tsv')
    ann_train = read_annotations('ner0531/train_tweet_annotations5.31.tsv')
    return tweets_all, ann_train, ann_test

def read_download():
    tweets_all = read_ann_tweets('download_tweets/data_all.txt')
    ann_test = read_annotations('download_tweets/test_tweet_annotations.tsv')
    ann_train = read_annotations('download_tweets/train_tweet_annotations.tsv')
    return tweets_all, ann_train, ann_test

def align_data(tweets_all, ann_train, ann_test):
    aligned_test = align_tweets_annotations(tweets_all, ann_test)
    aligned_train = align_tweets_annotations(tweets_all, ann_train)
    print('train: ', len(aligned_train), ' test: ', len(aligned_test))
    test = mark_annotations(aligned_test)
    train = mark_annotations(aligned_train)
    show_marked(train[1])
    show_marked(test[2])
    return train, test

def save_orig(train, test, basedir):
    writer = ConllWriter(basedir + '/orig/train.txt')
    writer(train)
    writer = ConllWriter(basedir + '/orig/test.txt')
    writer(test)

def save_splits(train, test, basedir):
    fold5_split(train, test, basedir + '/split')
    
if __name__ == '__main__':
    basedir = sys.argv[1]
    # read tweets and annotations
    # tweets_all, ann_train, ann_test = read_download()
    tweets_all, ann_train, ann_test = read_ner0531()
    # mark tweets base on annotations
    train, test = align_data(tweets_all, ann_train, ann_test)
    # save the result
    save_orig(train, test, basedir)
    save_splits(train, test, basedir)
