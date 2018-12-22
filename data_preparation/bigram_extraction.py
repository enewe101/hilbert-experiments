import os
import sys
import random
import codecs
import argparse
import itertools
from collections import Counter
import time

import numpy as np

import hilbert as h
import shared
import data_preparation as dp
from multiprocessing import Pool


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


def extract_unigram_parallel(
    corpus_path, num_workers, save_path=None, verbose=True
):
    pool = Pool(num_workers)
    args = (
        (corpus_path, worker_id, num_workers, verbose) 
        for worker_id in range(num_workers)
    )
    unigrams = pool.map(extract_unigram_parallel_worker, args)
    unigram = sum(unigrams, h.unigram.Unigram())
    if save_path is not None:
        unigram.save(save_path)
    return unigram


def extract_unigram_parallel_worker(args):
    corpus_path, worker_id, num_workers, verbose = args
    unigram = h.unigram.Unigram()
    file_chunk = dp.file_access.open_chunk(corpus_path, worker_id, num_workers)
    for line_num, line in enumerate(file_chunk):
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            print(line_num)
        tokens = line.strip().split()
        for token in tokens:
            unigram.add(token)
    return unigram


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



def extract_bigram_parallel(
    corpus_path, num_workers, unigram, sampler_type, window, min_count, thresh,
    save_path=None, verbose=True
):
    pool = Pool(num_workers)
    args = (
        (
            corpus_path, worker_id, num_workers, unigram,
            sampler_type, window, min_count, thresh, verbose
        ) 
        for worker_id in range(num_workers)
    )
    bigrams = pool.map(extract_bigram_parallel_worker, args)

    merged_bigram = bigrams[0]
    for bigram in bigrams[1:]:
        merged_bigram.merge(bigram)

    if save_path is not None:
        merged_bigram.save(save_path)
    return merged_bigram



def extract_bigram_parallel_worker(args):
    (
        corpus_path, worker_id, num_workers, unigram, sampler_type, 
        window, min_count, thresh, verbose
    ) = args
    bigram = h.bigram.Bigram(unigram)
    sampler = dp.bigram_sampler.get_sampler(
        sampler_type, bigram, window, min_count, thresh)
    file_chunk = dp.file_access.open_chunk(corpus_path, worker_id, num_workers)
    start = time.time()
    for line_num, line in enumerate(file_chunk):
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            print('elapsed', time.time() - start)
            start = time.time()
            print(line_num)
        sampler.sample(line.split())

    return bigram



def extract_unigram_and_bigram(
    corpus_path,
    out_dir,
    sampler_type,
    window,
    processes=1,
    min_count=None,
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
        if vocab is not None and len(unigram) > vocab:
            raise ValueError(
                'An existing unigram object was found on disk, having a '
                'vocabulary size of {}, but a vocabulary size of {} was '
                'requested.  Either truncate it manually, or run extraction '
                'for existing vocabulary size.'.format(len(unigram), vocab)
            )
        elif min_count is not None and min(unigram.Nx) < min_count:
            raise ValueError(
                'An existing unigram object was found on disk, containing '
                'tokens occuring only {} times (less than the requested '
                'min_count of {}).  Either prune it manually, or run '
                'extraction with `min_count` reduced.'.format(
                    min(unigram.Nx), min_count))

    except IOError:
        print('None found.  Training unigram data...')
        unigram = extract_unigram_parallel(corpus_path, processes)
        if vocab is not None:
            unigram.truncate(vocab)
        if min_count is not None:
            unigram.prune(min_count)

        print('Saving unigram data...')
        unigram.save(out_dir)

    # Train the bigram, and save it to disc.
    print('Training bigram data...')
    bigram = extract_bigram_parallel(
        corpus_path, processes, unigram, sampler_type, window, min_count, 
        thresh
    )
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
            "for use with w2v sampler only"
        )
    )
    parser.add_argument(
        '--sampler', '-s', help="Type of sampler to use",
        choices=('w2v', 'flat', 'harmonic', 'dynamic'), required=True,
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
    parser.add_argument(
        '--processes', '-p', help="Number of processes to spawn",
        default=1, type=int
    )
    parser.add_argument(
        '--min-count', '-m', default=None, type=int,
        help="Minimum number of occurrences below which token is ignored",
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

    if args['min_count'] is not None and args['vocab'] is not None:
        raise ValueError('Use either --vocab or --min-count, not both.')

    # thresh should only be specified if the sampler is w2v
    if args['thresh'] is not None and args['sampler_type'] != 'w2v':
        raise ValueError('thresh argument is only valid for w2v sampler.')

    extract_unigram_and_bigram(**args)

