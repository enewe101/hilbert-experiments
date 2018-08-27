import data_preparation as prep
import codecs
from scipy import sparse
import os
from collections import Counter
import numpy as np



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

        


def extract_cooccurrence_from_file(path, window, dictionary, counter):
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
    sparse.save_npz(path + '.Nxx.npz', N_xx)
    np.savez(path + '.Nx.npz', N_x)
    dictionary.save(path + '.dictionary')


def load_cooccurrence(path):
    return (
        sparse.load_npz(path + '.Nxx.npz'),
        np.load(path + '.Nx.npz')['arr_0'],
        Dictionary.load(path + '.dictionary')
    )


def extract_all(in_paths, window, limit=None):

    dictionary = Dictionary()
    counter = Counter()

    if limit is None:
        total = len(in_paths)
    else:
        total = min(len(in_paths), limit)

    for i, in_path in enumerate(in_paths):
        if limit is not None and i >= limit:
            break
        print '%.2f%%' % (float(i)/total), os.path.basename(in_path)
        extract_cooccurrence_from_file(in_path, window, dictionary, counter)

    N_xx = dict_to_sparse(counter)
    N_x = np.array(np.sum(N_xx, axis=1)).reshape(-1)

    N_xx, N_x, dictionary = sort_cooccurrence(N_xx, N_x, dictionary)

    return N_xx, N_x, dictionary


def extract_and_write_all(in_paths, out_path, window, limit=None):
    N_xx, N_x, dictionary = extract_all(in_paths, window, limit)
    save_cooccurrence(out_path, N_xx, N_xx, dictionary)
    return N_xx, N_x, dictionary
    

