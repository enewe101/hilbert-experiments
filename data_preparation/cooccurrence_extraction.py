import os
import sys
import codecs
import itertools
from collections import Counter

import numpy as np

import hilbert as h
import data_preparation as dp


def read_stats(name):
    path = dp.path_iteration.get_cooccurrence_path(name)
    return h.cooc_stats.read_stats(path)


# This is helpful if cooccurrence statistics have been gathered for shards
# (which are 1/16ths of gigaword).  But, a simpler approach is to gather
# cooccurrence statistics on all of gigaword at once, by using the large
# concatenation of all articles in gigaword, which has one document per line.
def accumulate_shards(name, truncate_before_adding=None):
    """
    name: Name of the particular cooccurrence statistics.  This is the name
        of the directory under /DATA_DIR/cooccurrence where the data is stored.
        it is also the suffix appended to name of the directory storing the 
        cooccurrence data for each shard of that dataset.

    truncate_before_adding: Use None for no truncation, or provide an integer to
        truncate each shard to that many of the most frequent words in each
        shard before accumulating them.  (Helps relieve the burden on memory
        for lower-memory environments.) The final running tally will have a bit
        more than this number in its vocabulary, because vocabularies of the
        shards are non-overlapping.
    """

    cooccurrences = h.cooc_stats.CoocStats()

    print('accumulating statistics in %s' % name)

    for i in ''.join([str(k) for k in range(10)]) + 'abcdef':

        dirname = '{name}/{i}00-{i}ff-{name}'.format(name=name, i=i)
        print('reading %s' % dirname)

        path = dp.path_iteration.get_cooccurrence_path(dirname)
        stats_to_add = h.cooc_stats.read_stats(path)
        if truncate_before_adding is not None:
            stats_to_add.truncate(truncate_before_adding)
        print('adding %s' % dirname)
        cooccurrences += stats_to_add

    out_dirname = '{name}/all-{name}'.format(name=name)
    out_path = dp.path_iteration.get_cooccurrence_path(out_dirname)
    cooccurrences.save(out_path)


def extract_cooccurrence_from_file(path, window, cooccurrences):
    """
    Extracts cooccurrence statistics, assuming that the file represents a
    single document, and counts cooccurrence relationships that span sentences,
    newlines, and paragraphs etc.
    """
    with codecs.open(path, 'r', 'utf8') as f:
        tokens = f.read().split()
    for i in range(len(tokens)):
        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(tokens):
                continue
            cooccurrences.add(tokens[i], tokens[j])



def extract_cooccurrence_from_file_per_line(
    path,
    window,
    cooccurrences,
    weighting_scheme='flat',
):
    """
    Extracts cooccurrence statistics, so that cooccurrence is only considered 
    to occur within each line.  This makes it possible to place separate
    documents on separate lines, and collect cooccurrences only within each
    document, while having only one large input file.  Alternatively, it 
    could be used to only sample cooccurrence within paragraphs or sentences,
    if the document is preprocessed to put these structural elements on their
    own lines.
    """
    weight = get_weight_getter(weighting_scheme)
    for with codecs.open(path, 'r', 'utf8') as in_file:
        for line in in_file:
            tokens = line.split()
            for i in range(len(tokens)):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                cooccurrence.add(tokens[i], tokens[j], count=weight(j, window))



def get_weight_getter(weighting_scheme):
    if weighting_scheme == 'flat':
        def weight(j, window_size):
            return 1
    elif weighting_scheme == 'dynamic':
        def weight(j, window_size):
            return 1./j
    elif weighting_scheme == 'harmonic':
        def weight(j, window_size):
            return float(window_size - j) / window_size
    return weight




def extract_all(in_paths, window, limit=None, verbose=True):
    cooccurrences = h.cooc_stats.CoocStats(verbose=verbose)
    if not isinstance(in_paths, list):
        in_paths = list(in_paths)
    if limit is None:
        total = len(in_paths)
    else:
        total = min(len(in_paths), limit)
    for i, in_path in enumerate(in_paths):
        if limit is not None and i >= limit:
            break
        if verbose and i%10==0:
            print('%.2f%%' % (100.0 * i / total), os.path.basename(in_path))
        extract_cooccurrence_from_file(in_path, window, cooccurrences)
    return cooccurrences


def extract_and_write_all(in_paths, out_path, window, limit=None, verbose=True):
    cooccurrences = extract_all(in_paths, window, limit, verbose)
    cooccurrences.save(out_path)
    return cooccurrences
    

def extract_sectors(sectors, out_path, window, limit=None):
    """
    Extracts cooccurrences for all files found in the provided sectors.  
    `sectors` should be a list of subdirs in gigaword-corenlp that are to
    be extracted.
    """
    print('using window size: ' + repr(window))
    print('writing to: ' + repr(out_path))
    print('limit: ' + repr(limit))
    in_paths = []
    for sector in sectors:
        print('extract_sectors: collecting paths for sector ' + sector)
        in_paths.extend(
            dp.path_iteration.iter_tokenized_paths_in_sector(sector))
    extract_and_write_all(in_paths, out_path, window, limit)


