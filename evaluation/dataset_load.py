from nltk.corpus import brown, conll2000
from os import listdir

SUPERVISED_DATASETS = {'brown': brown, 'conll2000': conll2000}
DS_DIR = 'datasets/'


def load_similarity():
    """
    Will return a dictionary. dataset_name -> [samples]
        where a sample is defined as tuple (str word1, str word2, float score)
        that is, word1 was voted to be "score" similar to word2
    """
    datasets = {}
    for fname in filter(lambda x: 'analog' not in x.lower(), listdir(DS_DIR)):
        with open('{}{}'.format(DS_DIR, fname), 'r') as f:
            ds = []
            for line in f.readlines()[1:]: # skip header
                strings = line.split(',')
                ds.append((*strings[:2], float(strings[2])))
            datasets[fname.split('.csv')[0]] = ds # rstrip had weird behavior
    return datasets


def load_analogies():
    """
    Will return a dictionary. dataset_name -> [samples]
        where a sample is defined as tuple (str word1, str word2, str word3, str word4)
        that is, word1 is to word2 as word3 is to word4
    """
    datasets = {}
    for fname in filter(lambda x: 'analog' in x.lower(), listdir(DS_DIR)):
        with open('{}{}'.format(DS_DIR, fname), 'r') as f:
            ds = [tuple(s.rstrip('\n') for s in l.split(',')) 
                  for l in f.readlines()[1:]]
            datasets[fname.split('.csv')[0]] = ds # rstrip had weird behavior
    return datasets


def load_pos_tagging(ds_str='brown'):
    ds = SUPERVISED_DATASETS[ds_str]


def load_chunking():
    ds = conll2000 


def load_sentiment():
    pass


def load_news_classification():
    pass



# main for basic testing
if __name__ == '__main__':
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
    


