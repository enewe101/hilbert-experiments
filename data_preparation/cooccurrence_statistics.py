import os
from itertools import izip
from collections import Counter

import numpy as np
from scipy import sparse

import data_preparation as dp




class CooccurrenceStatistics(object):


    def __init__(self, dictionary=None, counts=None, Nxx=None, verbose=True):
        '''
        Keeps track of token cooccurrences.  No arguments are needed to create
        an empty instance.  To create an instance that already contains counts
        supply (1) a dictionary and (2) either a 2-D numpy array of 
        CooccurrenceStatistics or a collections.Counter instance.

        dictionary (data_preparation.dictionary.Dictionary):
            A two-way mapping between tokens and ids.  Can be None if starting
            an empty CooccurrenceStatistics instance, otherwise required.

        counts (collections.Counter):
            Used to accumulates counts as a corpus is read.  Leave blank if
            starting an empty CooccurrenceStatistics.  Otherwise, it should have
            pairs (tuples) of token_ids as keys, and number of coocccurrences 
            as values.

        Nxx (numpy.array or scipy.sparse.csr matrix):
            Represents counts in a sparse format that is efficient for
            calculations, but not so convinient for accumulating counts as a
            corpus is read.
        '''

        self.validate_args(dictionary, counts, Nxx)

        if Nxx is None:
            self.basis = 'counts'
            self.synced = False
        else:
            self.basis = 'Nxx'
            self.synced = False

        self.dictionary = dictionary or dp.dictionary.Dictionary()

        self.counts = counts or Counter()
        if not isinstance(self.counts, Counter): 
            self.counts = Counter(self.counts)

        self._Nxx = Nxx
        Nxx_is_sparse_coo = isinstance(self._Nxx, sparse.coo_matrix)
        if self._Nxx is not None:
            if not Nxx_is_sparse_coo:
                self._Nxx = sparse.csr_matrix(self._Nxx)
            self._Nx = np.array(np.sum(self._Nxx, axis=1)).reshape(-1)
        else:
            self._Nx = None

        self.verbose = verbose
        self._denseNxx = None


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


    @property
    def Nxx(self):
        if self.basis == 'counts' and not self.synced:
            self.compile()
        return self._Nxx


    @property
    def Nx(self):
        if self.basis == 'counts' and not self.synced:
            self.compile()
        return self._Nx


    @property
    def denseNxx(self):
        if self._denseNxx is None:
            self._denseNxx = self.Nxx.todense()
        return self._denseNxx


    def add(self, token1, token2):
        if self.basis == 'Nxx':
            self.decompile()

        id1 = self.dictionary.add_token(token1)
        id2 = self.dictionary.add_token(token2)
        self.counts[id1, id2] += 1

        # We are no longer synced
        self.synced = False
        self._denseNxx = None


    def decompile(self, force=False):
        if self.basis == 'counts':
            raise ValueError(
                'Cannot decompile CooccurrenceStats: already uncompiled')
        if self.verbose:
            print 'decompiling cooccurrence stats...'

        Nxx_coo = self._Nxx.tocoo()
        self.counts = Counter()
        for i,j,v in izip(Nxx_coo.row, Nxx_coo.col, Nxx_coo.data):
            self.counts[i,j] = v

        self.basis = 'counts'
        self.synced = True


    def compile(self):
        if self.basis == 'Nxx':
            raise ValueError(
                'Cannot compile CooccurrenceStatistics: already compiled.')

        vocab_size = len(self.dictionary)
        print 'vocab size: ', vocab_size
        self._Nxx = dict_to_sparse(self.counts, (vocab_size,vocab_size))
        self._Nx = np.array(np.sum(self._Nxx, axis=1)).reshape(-1)
        self.sort(True)

        self.synced = True
        self._denseNxx = None


    def sort(self, force=False):
        if not force and (self.basis == 'counts' and not self.synced):
            self.compile()
        top_indices = np.argsort(-self._Nx.reshape((-1,)))
        self._Nxx = self._Nxx[top_indices][:,top_indices]
        self._Nx = self._Nx[top_indices]
        self.dictionary = dp.dictionary.Dictionary([
            self.dictionary.tokens[i] for i in top_indices])
        index_map = {
            old_idx: new_idx 
            for new_idx, old_idx in enumerate(top_indices)
        }
        new_counts = Counter()
        for (i,j), count in self.counts.iteritems():
            new_counts[index_map[i], index_map[j]] = count
        self.counts = new_counts


    def save(self, path):
        if self.basis == 'counts' and not self.synced:
            self.compile()
        os.makedirs(path)
        sparse.save_npz(os.path.join(path, 'Nxx.npz'), self._Nxx)
        #np.savez(os.path.join(path, 'Nx.npz'), Nx)
        self.dictionary.save(os.path.join(path, 'dictionary'))


    def density(self, threshold_count=0):
        if self.basis == 'counts' and not self.synced:
            self.compile()
        num_cells = np.prod(self._Nxx.shape)
        num_filled = (
            self._Nxx.getnnz() if threshold_count == 0 
            else np.sum(self._Nxx>threshold_count)
        )
        return float(num_filled) / num_cells


    def truncate(self, k):
        if self.basis == 'counts' and not self.synced:
            self.compile()
        self._Nxx = Nxx[:k][:,:k]
        self.Nx = Nx[:k]
        dictionary = Dictionary(dictionary.tokens[:k])
        return Nxx, Nx, dictionary


    @staticmethod
    def load(path):
        return CooccurrenceStatistics(
            Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')),
            dictionary=dp.dictionary.Dictionary.load(
                os.path.join(path, 'dictionary'))
        )




def dict_to_sparse(d, shape=None):
    I, J, V = [], [], []
    for (idx1, idx2), value in d.iteritems():
        I.append(idx1)
        J.append(idx2)
        V.append(value)

    return sparse.coo_matrix((V,(I,J)), shape).tocsr()


