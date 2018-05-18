import random
import re



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
    all_words = {}
    for i, sample in enumerate(samples):
        sent = sample[3]
        for word in sent.split():
            k = word.lower()
            all_words.setdefault(k, set())
            all_words[k].add(i)
    return all_words

def expand_samples(all_words, samples):
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
