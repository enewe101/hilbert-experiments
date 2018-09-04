import os
from itertools import izip
from collections import Counter

import numpy
from scipy import sparse

import data_preparation as dp




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




