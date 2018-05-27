import sys
import random
import re
import string
import spacy
from spacy.tokenizer import Tokenizer


head_word = re.compile('^[A-Z]: ')

def extract_from_ade(filename, outfile):
    lines = [line.split('|') for line in open(filename).readlines()]
    sents = set(line[1] for line in lines)
    with open(outfile, 'w') as fp:
        for sent in sents:
            fp.write('000\t000\t1\t%s\n' % head_word.sub('', sent))


similars = [
    ['i', 'you', 'she', 'he', 'we', 'us', 'me'],
    ['this', 'that'],
    ['on', 'off', 'for', 'of'],
    ['is', 'was', 'be'],
]

def load_samples(filename):
    return [line.split('\t') for line in open(filename)]

def save_samples(samples, outfile):
    random.shuffle(samples)
    with open(outfile, 'w') as fp:
        for sample in samples:
            fp.write('\t'.join(sample))

def balance_samples(samples):
    '''
    balance positive and negative samples
    '''
    pos = []
    neg = []
    for i, s in enumerate(samples):
        if s[2] == '0':
            neg.append(i)
        else:
            pos.append(i)
    pos.extend(neg[:len(pos)])
    balanced = []
    for t in pos:
        balanced.append(samples[t])
    return balanced

def count_words(samples):
    '''
    count words in all samples
    '''
    all_words = {}
    for i, sample in enumerate(samples):
        sent = sample[3]
        for word in sent.split():
            k = word.lower()
            all_words.setdefault(k, set())
            all_words[k].add(i)
    return all_words

def expand_samples(all_words, samples):
    '''
    expand samples by replacing similar words from a dictionary
    '''
    for word in all_words.keys():
        for sim in similars:
            if word not in sim:
                continue
            for sid in all_words[word]:
                # generate new positive sample by replacing with all other similar words
                if samples[sid][2] == '0':
                    continue
                sent = samples[sid][3].lower()
                orig_start = sent.find(word)
                orig_word = samples[sid][3][orig_start : orig_start + len(word)]
                for other in sim:
                    if other == word:
                        continue
                    new_sample = [samples[sid][0], samples[sid][1], samples[sid][2], samples[sid][3].replace(orig_word, other)]
                    print(samples[sid], '\n\t', new_sample)
                    samples.append(new_sample)

def filter_ner(infile, outfile):
    '''
    extract NER samples from merge_train and merge_test
    '''
    writer = open(outfile, 'w')
    with open(infile) as fp:
        for line in fp:
            items = line.replace('/', '').split()
            if len(items) < 4:
                continue
            tags = items[2].split('-')
            if len(tags) < 2:
                continue
            if items[3] == 'not' and items[4] == 'exist....':
                continue
            annotation = []
            for w in items[3:]:
                punc = ''
                if w[-1] in [',', '.', ':', '!']:
                    punc = w[-1]
                    w = w[:-1]
                if w.lower() == tags[0]:
                    annotation.append(w + '/' + 'B-DRUG')
                else:
                    annotation.append(w + '/' + 'O')
                if punc != '':
                    annotation.append(punc + '/' + 'O')
            print(' '.join(annotation))
            writer.write('\t'.join(annotation) + '\n')
    writer.close()

def read_tweets(infiles):
    tweets = {}
    # load all tweets
    for f in infiles:
        with open(f) as fp:
            for line in fp:
                items = line.split('\t')
                if items[3].startswith('not exist'):
                    continue
                tweets[items[2]] = [items[3].lower().strip()]
    return tweets


def rewrite_nickname(sample):
    '''
    @nick can be rewritten in many forms without changing the meaning
    '''
    nicks = []
    for word in sample.split():
        if word.startswith('@'):
            nicks.append(word)

    # generate 3 different name for each nickname
    new_samples = []
    for name in nicks:
        for _ in range(3):
            randstr = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            sent = sample.replace(name, '@' + randstr)
            new_samples.append(['888', '888', sample[2], sent])
    return new_samples

def locate_spans(sent, phrases):
    '''
    find all locations of phrases in a sentence
    '''
    spans = []
    for phrase in phrases:
        last_end = 0
        while last_end < len(sent):
            start = sent.find(phrase, last_end)
            if start < 0:
                break
            last_end = start + len(phrase)
            spans.append((start, last_end))
    return sorted(spans)


def update_annotations(tweets, ann_file):
    '''
    align annotations file with tweets
    the start and end in annotation files may not match with tweet files
    '''
    # load all annotations
    annotations = {}
    with open(ann_file) as fp:
        for line in fp:
            tid, start, end, semantic, span, drug1, drug2 = line.split('\t')
            span = span.lower()
            if tweets.get(tid) is None or semantic != 'ADR':
                continue

            annotations[tid] = []
            for tweet in tweets.get(tid):
                annotations[tid].append(locate_spans(tweet, [span, drug1]))
    return annotations


def tag_span(tokenizer, text, tag):
    '''
    convert all tokens in text with a specific tag
    '''
    tags = []
    if tag == 'O':
        for tok in tokenizer(text):
            # check "..."
            if tok.text.find('...') >= 0:
                start = tok.text.find('...')
                if start > 0:
                    tags.append(tok.text[:start] + '/O')
                tags.append('.../O')
                if start + 3 < len(tok.text):
                    tags.append(tok.text[start + 3:] + '/O')
            # check punctations
            elif tok.text[-1] in [',', '.', '!', ':', '~']:
                if len(tok.text) > 1:
                    tags.append(tok.text[:-1] + '/O')
                tags.append(tok.text[-1] + '/O')
            else:
                tags.append(tok.text + '/O')
    else:
        for tok in tokenizer(text):
            if len(tags) < 1:
                tags.append(tok.text + '/B-' + tag)
            else:
                tags.append(tok.text + '/I-' + tag)

    # finally, check empty token
    new_tags = []
    for t in tags:
        items = t.split('/')
        if len(items) < 2:
            continue
        if len(items[0]) < 1 or items[0].strip() == '':
            continue
        new_tags.append(t)
    return new_tags


def align_annotations(tweets, annotations):
    '''
    fetch raw tweet, then add tags based on annotations
    '''
    nlp = spacy.load('en_core_web_sm')
    tokenizer = Tokenizer(nlp.vocab)

    result = []
    for tid in annotations.keys():
        for i in range(len(tweets[tid])):
            text = tweets[tid][i]
            spans = annotations[tid][i]
            curr = 0
            last_end = 0
            aligned = []
            while curr < len(spans):
                start, end = spans[curr]
                aligned.extend(tag_span(tokenizer, text[last_end: start], 'O'))
                aligned.extend(tag_span(tokenizer, text[start: end], 'ADR'))
                last_end = end
                curr += 1
            if last_end < len(text):
                aligned.extend(tag_span(tokenizer, text[last_end:], 'O'))
            # print('\norig:', tweets[tid])
            # print(' '.join(aligned))
            result.append(aligned)
    return result


def annotate_tweets(orig_file, ann_file, out_file):
    '''
    read tweets and annotations from files and output IOB data
    '''
    tweets = read_tweets(orig_file)
    for tid in tweets:
        tweets[tid].extend(rewrite_nickname(tweets[tid][0]))

    annotations = update_annotations(tweets, ann_file)
    with open(out_file, 'w') as fp:
        all_data = list(align_annotations(tweets, annotations))
        for _ in range(5):
            random.shuffle(all_data)
        for aligned in all_data:
            fp.write(' '.join(aligned) + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python adr_util.py ann.tsv train.txt')
        sys.exit(0)
    ann_file = sys.argv[1]
    out_file = sys.argv[2]
    annotate_tweets(['data/orig_tweets/train_tweet.txt', 'data/orig_tweets/test_tweet.txt'],
                    ann_file, out_file)
