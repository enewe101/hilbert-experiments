import nltk
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nltk.corpus import brown, conll2000, ptb
from os import listdir
from evaluation.constants import *

UNSUP_DIR = 'unsup_datasets/'
SUP_DIR = 'sup_datasets/'

# filter helper function
def filter_words(tok_list, filter_stops):
    if filter_stops:
        return [t for t in tok_list if not t in STOPS]
    return tok_list



class HilbertDataset(object):
    """
    General class for holding our datasets for the different types.
    In the unsupervised case it will simply hold each of the many
    different datasets. In the supervised case it will hold a split
    of the train and test sets, allowing for CV over train set.
    """
    def __init__(self, name, is_unsupervised):
        self.name = name
        self.is_unsupervised = is_unsupervised
        self._train = []
        self._test = []
        self._all = {}
        self.labels_to_idx = {}


    def add_full(self, data_list, ds_name=''):
        assert not ds_name in self._all
        name_given = len(ds_name) > 0
        assert (self.is_unsupervised and name_given) or \
               (not self.is_unsupervised and not name_given)
        self._all[ds_name] = data_list


    def add_train(self, data_list):
        self._train += data_list


    def add_test(self, data_list):
        self._test += data_list
    

    def sup_check(self, ds_set='train'):
        assert not self.is_unsupervised
        assert len(self._train) > 0 
        assert len(self._test) > 0
        assert ds_set in ('train', 'test')


    def _ds_set(self, ds):
        return self._train if ds == 'train' else self._test


    def get_full(self, dname):
        assert self.is_unsupervised
        assert dname in self._all 
        return self._all[dname]


    def iter_ds_names(self):
        return self._all.keys()


    def items(self):
        return self._all.items()


    def values(self):
        return self._all.values()


    def split_train_test(self, proportion=0.3):
        assert len(self._train) == len(self._test) == 0
        assert not self.is_unsupervised
        data = self._all['']
        self._train, self._test = train_test_split(data, 
                                                   test_size=proportion,
                                                   random_state=1917)


    def get_x_y(self, ds_set,
                ds_dict=None,
                as_indicies=False,
                filter_stopwords=False,
                translate_label_by_one=False,
                ):
        self.sup_check(ds_set)
        label_translate = 1 if translate_label_by_one else 0
        set_indices = len(self.labels_to_idx) == 0 and as_indicies
        _x = []
        _y = []

        # storing the unique labels, if desired later
        unique_labs = set()

        # iterate over the tuples in the dataset
        for item in self._ds_set(ds_set):

            # if this is the case, then this is sequence classification
            if type(item[0]) == list and type(item[1]) == str: 
                sample, label = item
                _x.append(filter_words(sample, filter_stopwords))
                _y.append(label)

                # updating the unique labels for seq classificatoin
                if set_indices:
                    unique_labs.add(label)

            # otherwise, this is token-level seq-to-seq classification
            else:
                labels = [t[-1] for t in item]
                token_tups = [t[:-1][0] for t in item]
                _x.append(token_tups) # for POS tagging this will just be tokens
                                      # but for chunking it will be (token, tag)
                _y.append(labels)

                # updating the unique labels for seq-to-seq
                if set_indices:
                    unique_labs.update(set(labels))

        # set our label-to-idx storage if it has not been set before
        if set_indices:
            self.labels_to_idx = {l: i + label_translate for i, l in enumerate(unique_labs)}

        # return a list of indicies of labels, rather than the string labels themselves
        if as_indicies:
            toks_idx = []
            labs_idx = []
            for sentence, yval in zip(_x, _y):

                # get the idxs for the tokens from our dict
                toks_idx.append([])
                for t in sentence:

                    # TODO: fix this to account for the token tups for chunking
                    try:
                        t_id = ds_dict.get_id(t)
                    except KeyError:
                        t_id = len(ds_dict) # unk id
                    toks_idx[-1].append(t_id)

                # set the y labels
                if type(yval) == list:
                    labs_idx.append([self.labels_to_idx[lab] for lab in yval])
                    assert all(idx >= label_translate for idx in labs_idx[-1])  # all labels MUST start at 1 (padding is 0)
                else:
                    labs_idx.append(self.labels_to_idx[yval])

            # return all the indexed boys
            return toks_idx, labs_idx

        # otherwise, return the string labels
        else:
            return _x, _y


    def get_stats(self, ds_set):
        self.sup_check(ds_set)
        counts = defaultdict(lambda: 0)
        _, y = self.get_x_y(ds_set)
        for item in y:
            if type(item) == list:
                for val in item:
                    counts[val] += 1
            elif type(item) == str:
                counts[item] += 1
            else:
                print(item)
                raise NotImplementedError('Improper dataset form!')

        total = sum(counts.values())
        print('\nStats for dataset {} ({}):'.format(self.name, ds_set))
        for label, count in sorted(counts.items(), key=lambda t: t[1]):
            print('\t{:20}: {:6} samples ({:0.3f}% of total)'.format(
                    label, count, count / total * 100))
        print('\t{} TOTAL'.format(total))
        print('\t{}, length of Y'.format(len(y)))


    def get_unique_words_counts(self):
        vocab = defaultdict(lambda: 0)
        if self.is_unsupervised:
            for ds, samples in self.items():
                for sample in samples:
                    if type(sample) == str:
                        vocab[sample] += 1
        else:
            for ds in ('train', 'test'):
                x, y = self.get_x_y(ds)
                for sample in x:
                    if type(sample) == list:
                        for s in sample:
                            vocab[s] += 1
                    elif type(sample) == str:
                        vocab[sample] += 1
                    else:
                        raise NotImplementedError
        return vocab




def load_similarity():
    """
    Basically a dictionary dataset object. dataset_name -> [samples]
        where a sample is defined as tuple (str word1, str word2, float score)
        that is, word1 was voted to be "score" similar to word2
    """
    datasets = HilbertDataset('similarity', True) 
    for fname in filter(lambda x: 'analog' not in x.lower(), listdir(UNSUP_DIR)):
        with open('{}{}'.format(UNSUP_DIR, fname), 'r') as f:
            ds = []
            for line in f.readlines()[1:]: # skip header
                strings = line.split(',')
                ds.append((strings[0].lower(), 
                           strings[1].lower(), 
                           float(strings[2]),))
            datasets.add_full(ds, fname.replace('.csv', '')) 
    return datasets


def load_analogies():
    """
    Basically a dictionary dataset object. dataset_name -> [samples]
        where a sample is defined as tuple (str word1, str word2, str word3, str word4)
        that is, word1 is to word2 as word3 is to word4
    """
    datasets = HilbertDataset('analogy', True)
    for fname in filter(lambda x: 'analog' in x.lower(), listdir(UNSUP_DIR)):
        with open('{}{}'.format(UNSUP_DIR, fname), 'r') as f:
            ds = [tuple(s.rstrip('\n').lower() for s in l.split(',')) 
                  for l in f.readlines()[1:]]
            datasets.add_full(ds, fname.replace('.csv', '')) 
    return datasets


def load_brown_pos_tagging():
    """
    Returns a full list representing the dataset.
        Each item is a sentence built of (word, tag) tuples for each token.
    """
    dataset = HilbertDataset('brown-pos', is_unsupervised=False)
    sents_tags = [[(w.lower(), t) for w, t in s]
                  for s in brown.tagged_sents(tagset='universal')]
    dataset.add_full(sents_tags)
    return dataset


def load_wsj_pos_tagging():
    """
    Returns a full list representing the WSJ PTB2 dataset for POS-tagging.
        Each item is a sentence built of (word, tag) tuples for each token.
    """
    dataset = HilbertDataset('wsj-pos', is_unsupervised=False)
    files = list(ptb.fileids())
    test_dirs = ['22', '23', '24']
    train_d, test_d = [], []
    for f in files:
        data_list = test_d if f[4:6] in test_dirs else train_d
        data_list += [[(w.lower(), t) for w, t in s]
                       for s in ptb.tagged_sents(f)]
    dataset.add_train(train_d)
    dataset.add_test(test_d)
    return dataset


def load_chunking():
    """
    Returns a dictionary for the train and test sets.
        Each sample in the dictionary is a sentence.
            Each sentence contains a ((word, tag), chunk) 
            tuples for each token in the sentence.
    May want to look into using our model to build the POS tags first
        and then doing the chunking, a more realistic setting (perhaps).
    """
    dataset = HilbertDataset('chunking', is_unsupervised=False)
    for dname in ('train', 'test'):
        sents = conll2000.chunked_sents('{}.txt'.format(dname), chunk_types=['NP'])
        data = [ [((w.lower(), t), c) for w, t, c in nltk.chunk.tree2conlltags(sent)] 
                 for sent in sents ]
        dataset.add_train(data) if dname == 'train' else dataset.add_test(data)
    return dataset


def load_sentiment():
    dataset = HilbertDataset('sentiment', is_unsupervised=False)
    for fname in filter(lambda x: 'imdb' in x.lower(), listdir(SUP_DIR)):
        ds = []
        with open('{}{}'.format(SUP_DIR, fname), 'r') as f:
            for line in f.readlines()[1:]:
                splitted = line.split(',')
                assert len(splitted) == 2
                # (sample tokens, label)
                ds.append((splitted[0].split(' '), splitted[1].rstrip('\n')))
        assert len(ds) > 0
        dataset.add_train(ds) if 'train' in fname else dataset.add_test(ds)
    return dataset


def load_news_classification():
    dataset = HilbertDataset('news', is_unsupervised=False)
    ds = []
    with open('{}agnews_full.csv'.format(SUP_DIR), 'r') as f:
        for line in f.readlines()[1:]:
            splitted = line.split(',')
            assert len(splitted) == 2
            # (sample tokens, label)
            ds.append((splitted[0].split(' '), splitted[1].rstrip('\n')))
    assert len(ds) > 0
    dataset.add_full(ds)
    return dataset


def load_all():
    all_hilbs = [
        load_similarity(),
        load_analogies(),
        load_brown_pos_tagging(),
        load_wsj_pos_tagging(),
        load_chunking(),
        load_sentiment(),
        load_news_classification(),
    ]
    for d in all_hilbs:
        try:
            d.split_train_test()
        except AssertionError:
            pass
    return { d.name: d for d in all_hilbs }




# main for basic testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build datasets for hilbert.')
    parser.add_argument('--testing', action='store_true',
                        help='run tests over the datasets')
    args = parser.parse_args()

    if not args.testing:
        # save it all up
        all_data = load_all()
        np.savez_compressed('np/all_data.npz', np.array([all_data]))

    def test():

        class TestDictionary(object):
            def get_id(self, t):
                return 0


        # similarity ds tests
        sim_ds = load_similarity()
        for key, sample_list in sim_ds.items():
            print('{:5} samples in {:25}'.format(len(sample_list), key))
            for s in sample_list:
                assert len(s) == 3
                assert type(s) == tuple
                assert type(s[0]) == type(s[1]) == str
                assert type(s[2] == float)
            print('\tExample: {}'.format(sample_list[15]))
        print('Similarity tests passed.\n')

        # analogytests
        analogy_ds = load_analogies()
        for key, sample_list in analogy_ds.items():
            print('{:5} samples in {:25}'.format(len(sample_list), key))
            for s in sample_list:
                assert len(s) == 4
                assert type(s) == tuple
                assert all([type(st) == str for st in s])
            print('\tExample: {}'.format(sample_list[15]))
        print('Analogy tests passed.\n')

        # pos tags
        print('Brown Pos tagging')
        pos = load_brown_pos_tagging()
        pos.split_train_test()
        x, y = pos.get_x_y('train')
        print(x[0:2])
        print(y[0:2])
        pos.get_stats('train')
        # testing labelling
        x, y = pos.get_x_y('train', ds_dict=TestDictionary(),
                           as_indicies=True,
                           translate_label_by_one=True)
        uniques = set()
        for label_list in y:
            for label in label_list:
                uniques.add(label)
        for label_list in y:
            for label in label_list:
                assert 0 < label <= len(uniques)
        print()

        print('WSJ Pos tagging')
        pos = load_wsj_pos_tagging()
        x, y = pos.get_x_y('train')
        print(x[0:2])
        print(y[0:2])
        pos.get_stats('train')
        # testing labelling
        x, y = pos.get_x_y('train', ds_dict=TestDictionary(),
                           as_indicies=True,
                           translate_label_by_one=True)
        uniques = set()
        for label_list in y:
            for label in label_list:
                uniques.add(label)
        for label_list in y:
            for label in label_list:
                assert 0 < label <= len(uniques)
        print()
        
        # chunking
        # print('Chunking')
        # chunk = load_chunking()
        # x, y = chunk.get_x_y('train')
        # print(x[0:2])
        # print(y[0:2])
        # chunk.get_stats('train')
        # print()
        
        # sentiment
        print('Sentiment')
        sent = load_sentiment()
        x, y = sent.get_x_y('train', filter_stopwords=True)
        print(x[0:2])
        print(y[0:2])
        sent.get_stats('train')

        # testing labelling
        x, y = sent.get_x_y('train', ds_dict=TestDictionary(),
                            as_indicies=True,
                            translate_label_by_one=False)
        uniques = set(y)
        assert 0 in uniques
        for label in y:
            assert 0 <= label < len(uniques)
        print()

        # news
        print('News')
        news = load_news_classification()
        news.split_train_test()
        x, y = news.get_x_y('train', filter_stopwords=True)
        print(x[0:2])
        print(y[0:2])
        news.get_stats('train')

        # testing labelling
        x, y = sent.get_x_y('train', ds_dict=TestDictionary(),
                            as_indicies=True,
                            translate_label_by_one=False)
        uniques = set(y)
        assert 0 in uniques
        for label in y:
            assert 0 <= label < len(uniques)
        print()


    if args.testing:
        test()

