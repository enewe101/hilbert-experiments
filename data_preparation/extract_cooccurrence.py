import os
import sys
import codecs
from itertools import izip
from collections import Counter

from scipy import sparse
import numpy as np

import data_preparation as dp


class Dictionary(object):

    def __init__(self, tokens=None):
        self.tokens = []
        self.token_ids = {}
        if tokens is not None:
            for token in tokens:
                self.add_token(token)

    def get_id(self, token):
        return self.token_ids[token]

    def get_token(self, idx):
        return self.tokens[idx]

    def add_token(self, token):
        if token not in self.token_ids:
            idx = len(self.tokens)
            self.token_ids[token] = idx
            self.tokens.append(token)
            return idx
        return self.token_ids[token]

    def save(self, path):
        codecs.open(path, 'w', 'utf8').write('\n'.join(self.tokens))

    @staticmethod
    def load(path):
        dictionary = Dictionary()
        dictionary.tokens = open(path).read().split('\n')
        dictionary.token_ids = {
            token: idx 
            for idx, token in enumerate(dictionary.tokens)
        }
        return dictionary

        

class CooccurrenceStatistics(object):


    def __init__(self, dictionary=None, counts=None, Nxx=None):
        self.validate_args(dictionary, counts, Nxx)

        if Nxx is None:
            self.source_of_truth = 'counts'
        else:
            self.source_of_truth = 'Nxx'

        self.dictionary = dictionary or Dictionary()
        self.counts = counts or Counter()
        self.Nxx = Nxx
        self.dense_Nxx = None


    def validate_args(self, dictionary, counts, Nxx):
        if counts is not None and Nxx is not None:
            raise ValueError(
                'Non-empty CooccurrenceStatistics objects should be '
                'instantiated by providing statistics either as a sparse '
                'matrix (Nxx) or a counter (counts)---not both.'
            )

        if counts is not None or Nxx is not None:
            if dictionary is None:
                raise ValueError(
                    'A dictionary must be provided to create a non-empty '
                    'CooccurrenceStatistics object.'
                )


    def add(self, token1, token2):
        id1 = self.dictionary.add_token(token1)
        id2 = self.dictionary.add_token(token2)
        self.counts[id1, id2] += 1


    def uncompile1(self):
        Nxx_coo = self.Nxx.tocoo()
        self.counts = Counter()
        for i,j,v in izip(Nxx_coo.row, Nxx_coo.col, Nxx_coo.data):
            self.counts[i,j] = v


    def compile(self):
        self.Nxx = dict_to_sparse(self.counter)
        self.Nx = np.array(np.sum(self.Nxx, axis=1)).reshape(-1)
        self.sort()


    def sort(self, Nxx, Nx, dictionary):
        top_indices = np.argsort(-self.Nx.reshape((-1,)))
        self.Nxx = self.Nxx[top_indices][:,top_indices]
        self.Nx = Nx[top_indices]
        self.dictionary = Dictionary([
            self.dictionary.tokens[i] for i in top_indices])


    def save(self, path):
        os.makedirs(path)
        sparse.save_npz(os.path.join(path, 'Nxx.npz'), Nxx)
        np.savez(os.path.join(path, 'Nx.npz'), Nx)
        self.dictionary.save(os.path.join(path, 'dictionary'))


    def density(self, threshold_count=0):
        num_cells = np.prod(self.Nxx.shape)
        num_filled = (
            self.Nxx.getnnz() if threshold_count == 0 
            else np.sum(self.Nxx>threshold_count)
        )
        return float(num_filled) / num_cells


    def truncate(self, k):
        self.Nxx = Nxx[:k][:,:k]
        self.Nx = Nx[:k]
        dictionary = Dictionary(dictionary.tokens[:k])
        return Nxx, Nx, dictionary


    @staticmethod
    def load_cooccurrence(path):
        return CooccurrenceStatistics(
            Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')),
            dictionary=Dictionary.load(os.path.join(path, 'dictionary'))
        )




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



def dict_to_sparse(d):
    I, J, V = [], [], []
    for (idx1, idx2), value in d.iteritems():
        I.append(idx1)
        J.append(idx2)
        V.append(value)

    return sparse.coo_matrix((V,(I,J))).tocsr()


def sort_cooccurrence(N_xx, N_x, dictionary):
    top_indices = np.argsort(-N_x.reshape((-1,)))
    N_xx = N_xx[top_indices][:,top_indices]
    N_x = N_x[top_indices]
    dictionary = Dictionary([dictionary.tokens[i] for i in top_indices])
    return N_xx, N_x, dictionary


def truncate_cooccurrence(k, N_xx, N_x, dictionary):
    N_xx = N_xx[:k][:,:k]
    N_x = N_x[:k]
    dictionary = Dictionary(dictionary.tokens[:k])
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
        Dictionary.load(os.path.join(path, 'dictionary'))
    )


def extract_all(in_paths, window, limit=None, verbose=False):

    dictionary = Dictionary()
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

    N_xx = dict_to_sparse(counter)
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


