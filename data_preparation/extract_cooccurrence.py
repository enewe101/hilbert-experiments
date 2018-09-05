import os
import sys
import codecs
from collections import Counter

from scipy import sparse
import numpy as np

import data_preparation as dp


        


def density(N_xx, threshold_count=0):
    num_cells = np.prod(N_xx.shape)
    num_filled = (
        N_xx.getnnz() if threshold_count == 0 
        else np.sum(N_xx>threshold_count)
    )
    return float(num_filled) / num_cells



def extract_cooccurrence_from_file(
    path, window, dictionary, counter, verbose=True
):
    if verbose:
        print os.path.basename(path)
    tokens = codecs.open(path, 'r', 'utf8').read().split()
    for i in range(len(tokens)):
        focal_token = tokens[i]
        focal_id = dictionary.add_token(focal_token)
        for j in range(i - window, i + window + 1):
            if j == i or j < 0 or j >= len(tokens):
                continue
            context_token = tokens[j]
            context_id = dictionary.add_token(context_token)
            counter[focal_id,context_id] += 1

    return counter




def sort_cooccurrence(N_xx, N_x, dictionary):
    top_indices = np.argsort(-N_x.reshape((-1,)))
    N_xx = N_xx[top_indices][:,top_indices]
    N_x = N_x[top_indices]
    dictionary = dp.dictionary.Dictionary([
        dictionary.tokens[i] for i in top_indices])
    return N_xx, N_x, dictionary


def truncate_cooccurrence(k, N_xx, N_x, dictionary):
    N_xx = N_xx[:k][:,:k]
    N_x = N_x[:k]
    dictionary = dp.dictionary.Dictionary(dictionary.tokens[:k])
    return N_xx, N_x, dictionary


def save_cooccurrence(path, N_xx, N_x, dictionary):
    os.makedirs(path)
    
    sparse.save_npz(os.path.join(path, 'Nxx.npz'), N_xx)
    np.savez(os.path.join(path, 'Nx.npz'), N_x)
    dictionary.save(os.path.join(path, 'dictionary'))


def load_cooccurrence(path):
    return (
        sparse.load_npz(os.path.join(path, 'Nxx.npz')),
        np.load(os.path.join(path, 'Nx.npz'))['arr_0'],
        dp.dictionary.Dictionary.load(os.path.join(path, 'dictionary'))
    )


def extract_all(in_paths, window, limit=None, verbose=False):

    dictionary = dp.dictionary.Dictionary()
    counter = Counter()
    if not isinstance(in_paths, list):
        in_paths = list(in_paths)

    if limit is None:
        total = len(in_paths)
    else:
        total = min(len(in_paths), limit)

    for i, in_path in enumerate(in_paths):
        if limit is not None and i >= limit:
            break
        if verbose:
            print ('%.2f%%' % (100.0 * i / total)), os.path.basename(in_path)
        extract_cooccurrence_from_file(
            in_path, window, dictionary, counter, verbose)

    N_xx = dp.cooccurrence_statistics.dict_to_sparse(counter)
    N_x = np.array(np.sum(N_xx, axis=1)).reshape(-1)

    N_xx, N_x, dictionary = sort_cooccurrence(N_xx, N_x, dictionary)

    return N_xx, N_x, dictionary


def extract_and_write_all(in_paths, out_path, window, limit=None):
    N_xx, N_x, dictionary = extract_all(in_paths, window, limit)
    save_cooccurrence(out_path, N_xx, N_xx, dictionary)
    return N_xx, N_x, dictionary
    

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


