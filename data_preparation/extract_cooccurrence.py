import data_preparation as prep
import codecs
from scipy import sparse
import os
from collections import Counter



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


def save_cooccurrences(path, N_xx, dictionary):
    sparse.save_npz(path + '.npz', N_xx)
    dictionary.save(path + '.dictionary')


def load_cooccurrences(path):
    return (
        sparse.load_npz(path + '.npz'),
        Dictionary.load(path + '.dictionary')
    )


def extract_all(write_path, window, limit=None):

    dictionary = Dictionary()
    counter = Counter()

    tokenized_fnames = list(prep.path_iteration.iter_tokenized_fnames())
    if limit is None:
        total = len(tokenized_fnames)
    else:
        total = min(len(tokenized_fnames), limit)

    for i, (dirname, in_fname) in enumerate(tokenized_fnames):
        if limit is not None and i >= limit:
            break
        print '%.2f%%' % (float(i)/total), in_fname
        in_path = prep.path_iteration.get_tokenized_path(dirname, in_fname)
        extract_cooccurrence_from_file(in_path, window, dictionary, counter)

    N_xx = dict_to_sparse(counter)

    return N_xx, dictionary


def extract_and_write_all(write_path, window, limit=None):
    N_xx, dictionary = extract_all(write_path, window, limit)
    save_cooccurrences(write_path, N_xx, dictionary)
    return N_xx, dictionary
    
    


