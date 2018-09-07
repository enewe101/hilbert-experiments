import os
import sys
import codecs
from collections import Counter

from scipy import sparse
import numpy as np

import hilbert as h
import data_preparation as dp



def extract_cooccurrence_from_file(path, window, cooccurrences):
    with codecs.open(path, 'r', 'utf8') as f:
        tokens = f.read().split()
    for i in range(len(tokens)):
        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(tokens):
                continue
            cooccurrences.add(tokens[i], tokens[j])



def extract_all(in_paths, window, limit=None, verbose=True):
    cooccurrences = h.cooc_stats.CooccurrenceStatistics(verbose=verbose)
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
    in_paths = (
        fname for sector in sectors 
        for fname in dp.path_iteration.iter_tokenized_paths_in_sector(sector)
    )
    extract_and_write_all(in_paths, out_path, window, limit)


STANDARD_WINDOW_SIZE = 5
if __name__ == '__main__':
    sectors = sys.argv[1:]
    dp.path_iteration.raise_if_sectors_not_all_valid(sectors)
    out_dir_name = '%s-%s-%dw' % (sectors[0], sectors[-1], STANDARD_WINDOW_SIZE)
    out_path = dp.path_iteration.get_cooccurrence_path(out_dir_name)
    extract_sectors(sectors, out_path, STANDARD_WINDOW_SIZE)


