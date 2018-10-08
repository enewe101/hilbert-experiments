import os
import sys
import random
import codecs
import itertools
from collections import Counter

import numpy as np

import hilbert as h
import shared
import data_preparation as dp


def read_stats(name):
    path = dp.path_iteration.get_cooccurrence_path(name)
    return h.cooc_stats.read_stats(path)


def read_test_stats():
    return read_stats('4w-flat/top10k-4w')


def extract_unigram(corpus_path, unigram, verbose=True):
    with open(corpus_path) as corpus_f:
        for line_num, line in enumerate(corpus_f):
            if verbose and line_num % 1000 == 0:
                print(line_num)
            tokens = line.strip().split()
            for token in tokens:
                unigram.add(token)


def extract_bigram_w2v_trace(in_path):
    positives, negatives = Counter(), Counter()
    epochs = 0
    print("Tallying...")
    with codecs.open(in_path, 'r', 'utf8') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num % 100000 == 0:
                print(line_num)
            if line.startswith('Epoch'):
                epochs += 1
                print("Epoch {}".format(epochs))
            fields = line.strip().split('\t')
            if len(fields) != 3:
                continue
            token1, token2, is_positive = fields
            is_positive = int(is_positive)
            if is_positive:
                positives[token1, token2] += 1
            else:
                negatives[token1, token2] += 1

    print("Normalizing...")
    if epochs > 1:
        for key in positives:
            positives[key] = positives[key] / epochs
        for key in negatives:
            negatives[key] = negatives[key] / epochs

    return positives, negatives



def extract_bigram(corpus_path, sampler, verbose=True):
    """
    Extracts cooccurrence statistics, so that cooccurrence is only considered 
    to occur within each line.  This makes it possible to place separate
    documents on separate lines, and collect cooccurrences only within each
    document, while having only one large input file.  Alternatively, it 
    could be used to only sample cooccurrence within paragraphs or sentences,
    if the document is preprocessed to put these structural elements on their
    own lines.
    """
    with codecs.open(corpus_path, 'r', 'utf8') as in_file:
        for line_num, line in enumerate(in_file):
            #if verbose and line_num % 1000 == 0:
            print(line_num)
            sampler.sample(line.split())


if __name__ == '__main__':

    corpus_fname = sys.argv[1]
    corpus_path = os.path.join(shared.CONSTANTS.TOKENIZED_CAT_DIR, corpus_fname)

    out_dirname = sys.argv[2]
    out_dirpath = os.path.join(shared.CONSTANTS.COOCCURRENCE_DIR, out_dirname)

    sampler_type = 'w2v'
    window = 5
    min_count = 1

    print('Corpus path:\t{}\n'.format(corpus_path))
    print('Output path:\t{}\n'.format(out_dirpath))
    print('Sampler type:\t{}\n'.format(sampler_type))
    print('Window size:\t{}\n'.format(window))

    # Attempt to read unigram, if none exists, then train it and save to disc.
    try:
        print('Attempting to read unigram data...')
        unigram = h.unigram.Unigram.load(out_dirpath)
    except IOError:
        print('None found.  Training unigram data...')
        unigram = h.unigram.Unigram()
        extract_unigram(corpus_path, unigram)
        print('Saving unigram data...')
        unigram.save(out_dirpath)

    # Train the bigram, and save it to disc.
    bigram = h.bigram.Bigram(unigram)
    sampler = dp.bigram_sampler.get_sampler(
        'w2v', bigram=bigram, window=window, thresh=1, min_count=min_count
    )
    print('Training bigram data...')
    extract_bigram(corpus_path, sampler)
    print('Saving bigram data...')
    bigram.save(out_dirpath)
