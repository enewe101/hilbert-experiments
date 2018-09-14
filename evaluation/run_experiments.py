import argparse
import hilbert
import numpy as np
import sklearn_crfsuite as skcrf
from nltk.corpus import stopwords
from dataset_load import HilbertDataset
from sklearn.linear_model import LogisticRegression

# global constants
EMB_DIM = 300



def load_embeddings(path, vocab):
    if path == 'RANDOM':
        words = [t[0] for t in vocab if t[1] > 1] # don't get single occurences
        words_d = hilbert.dictionary.Dictionary(words)
        return hilbert.embeddings.random(EMB_DIM, len(words), words_d)
    else:
        raise NotImplementedError


def get_all_words(list_of_hdatasets):
    vocab = {}
    for d in list_of_hdatasets:
        vocab.update(d.get_unique_words_counts())
    return list(sorted(vocab.items(), key=lambda t: t[1]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiments on set of embeddings.'
    )
    parser.add_argument('--emb_path', type=str, default='RANDOM',
        help='path to the embeddings we want to process,'
             'default is just using a constant random init.'
    )
    #parser.add_argument('')
    args = parser.parse_args() 

    # load up the datasets and get the vocab we will need.
    datasets = np.load('np/all_data.npz')['arr_0'][0]
    vocab_tups = get_all_words(datasets.values())
    emb = load_embeddings(args.emb_path, vocab_tups)

    # run the experiments!

