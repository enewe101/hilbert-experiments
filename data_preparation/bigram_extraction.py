import os
import sys
import random
import codecs
import argparse
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
    unigram.sort()


def extract_trace(in_path):
    """
    Given the trace output from running a modified form of word2vec, which
    contains each of the word-pair positive and negative samples, 
    extract bigram statistics based on the trace, and store them like a 
    hilbert.bigram and hilbert.unigram saved file.
    This is an alternative to directly extracting statistics from the corpus, 
    and is done in order to obtain the exact same cooccurrence statistics used 
    by word2vec can be used in hilbert models.
    """
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


def extract_trace_as_bigrams(trace_path, dictionary_path, out_dir):
    """
    Given the trace output from running a modified form of word2vec, which
    contains each of the word-pair positive and negative samples, 
    extract bigram statistics based on the trace, and store them like a 
    hilbert.bigram and hilbert.unigram saved file.
    This is an alternative to directly extracting statistics from the corpus, 
    and is done in order to obtain the exact same cooccurrence statistics used 
    by word2vec can be used in hilbert models.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    Nxx = np.zeros((len(dictionary), len(dictionary)))
    Nxx_neg = np.zeros((len(dictionary), len(dictionary)))
    epochs = 0
    with codecs.open(trace_path, 'r', 'utf8') as in_file:
        for line_num, line in enumerate(in_file):
            if line.startswith('Epoch'):
                epochs += 1
                print("Epoch {}".format(epochs))
                continue
            fields = line.strip().split('\t')
            if len(fields) != 3:
                continue
            token1, token2, is_positive = fields
            is_positive = int(is_positive)
            idx1, idx2 = dictionary.get_id(token1), dictionary.get_id(token2)
            if is_positive:
                Nxx[idx1, idx2] += 1
            else:
                Nxx_neg[idx1, idx2] += 1

    Nxx_out_path = os.path.join(out_dir, 'Nxx.npy')
    np.save(Nxx_out_path, Nxx)
    Nxx_neg_out_path = os.path.join(out_dir, 'Nxx_neg.npy')
    np.save(Nxx_neg_out_path, Nxx_neg)





def extract_bigram(corpus_path, sampler, verbose=True):
    """
    Extracts cooccurrence statistics, so that cooccurrence is only considered 
    to occur within each line.  This makes it possible to place separate
    documents on separate lines, and collect cooccurrences only within each
    document, while having only one large input file.  Alternatively, it 
    could be used to only sample cooccurrence within paragraphs or sentences,
    provided the document is preprocessed to put these structural elements on
    their own lines.
    """
    with codecs.open(corpus_path, 'r', 'utf8') as in_file:
        for line_num, line in enumerate(in_file):
            if verbose and line_num % 1000 == 0:
                print(line_num)
            sampler.sample(line.split())


def extract_unigram_and_bigram(
    corpus_path,
    out_dir,
    sampler_type,
    window,
    thresh=None,
    vocab=None
):

    print('Corpus path:\t{}\n'.format(corpus_path))
    print('Output path:\t{}\n'.format(out_dir))
    print('Sampler type:\t{}\n'.format(sampler_type))
    print('Window size:\t{}\n'.format(window))

    # Attempt to read unigram, if none exists, then train it and save to disc.
    try:
        print('Attempting to read unigram data...')
        unigram = h.unigram.Unigram.load(out_dir)
    except IOError:
        print('None found.  Training unigram data...')
        unigram = h.unigram.Unigram()
        extract_unigram(corpus_path, unigram)
        if vocab is not None:
            unigram.truncate(vocab)

        print('Saving unigram data...')
        unigram.save(out_dir)

    # Train the bigram, and save it to disc.
    bigram = h.bigram.Bigram(unigram)
    sampler = dp.bigram_sampler.get_sampler(sampler_type, bigram,window,thresh)
    print('Training bigram data...')
    extract_bigram(corpus_path, sampler)
    print('Saving bigram data...')
    bigram.save(out_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Extracts unigram and bigram statistics from a corpus, "
        "and stores to disk"
    ))
    parser.add_argument(
        '--corpus', '-c', required=True, dest='corpus_filename',
        help="File name for input corpus"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, 
        help="Name of directory in which to store cooccurrence data"
    )
    parser.add_argument(
        '--thresh', '-t', type=float, help=(
            "Threshold for common-word undersampling, "
            "for use with dynamic sampler only"
        )
    )
    parser.add_argument(
        '--sampler', '-s', help="Type of sampler to use",
        choices=('w2v', 'flat', 'harmonic'), required=True,
        dest="sampler_type"
    )
    parser.add_argument(
        '--window', '-w', help="Cooccurrence window size",
        required=True, type=int
    )
    parser.add_argument(
        '--vocab', '-v', type=int, default=None,
        help="Prune vocabulary to the most common `vocab` number of words"
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Corpus path and output directory are relative to standard locations.
    args['corpus_path'] = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['corpus_filename']
    )
    del args['corpus_filename']
    args['out_dir'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['out_dir']
    )
    

    # thresh should only be specified if the sampler is w2v
    if args['thresh'] is not None and args['sampler_type'] != 'w2v':
        raise ValueError('thresh argument is only valid for w2v sampler.')

    extract_unigram_and_bigram(**args)

