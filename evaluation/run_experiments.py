import argparse
import hilbert
import numpy as np
import sklearn_crfsuite as skcrf
from nltk.corpus import stopwords
from dataset_load import HilbertDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from progress.bar import IncrementalBar


# little helper
def cossim(v1, v2):
    dot = v1.dot(v2)
    return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))


# global constants
EMB_DIM = 300


def similarity_exp(embs, hdataset):
    results = {}

    # for showing over time
    total_iter = sum(len(samples) for samples in hdataset.values())
    bar = IncrementalBar('Running similarity experiments', max=total_iter)

    # iterate over all the similarity datasets in the object
    for dname, samples in hdataset.items():
        similarities = []
        gold = []
        for w1, w2, gold_score in samples:
            bar.next()

            gold.append(gold_score)
            e1 = embs.get_vec(w1, oov_policy='unk').numpy()
            e2 = embs.get_vec(w2, oov_policy='unk').numpy()
            similarities.append(cossim(e1, e2))
        
        results[dname] = spearmanr(gold, similarities)[0]
    
    bar.finish()
    return results


def analogy_exp(embs, hdataset):
    results = {}

    # normalize for faster sim calcs.
    embs.normalize()
    
    # for showing over time
    total_iter = sum(len(samples) for samples in hdataset.values())
    bar = IncrementalBar('Running analogy experiments', max=total_iter)
   
    # iterate over the two analogy datasets
    for dname, samples in hdataset.items():
        correct_cosadd = 0
        correct_cosmul = 0

        for w1, w2, w3, w4 in samples:
            bar.next()
            id1 = embs.dictionary.get_id(w1)
            id2 = embs.dictionary.get_id(w2)
            id3 = embs.dictionary.get_id(w3)

            # w1 is to w2 as w3 is to _argmax_
            # have to get two results, 3cosadd and 3cosmul
            best_add, best_mul = -np.inf, -np.inf
            best_w_add, best_w_mul = '', ''

            for i in range(len(embs.dictionary)):
                if i in (id1, id2, id3): continue
                sim1w = embs.V[i].dot(embs.V[id1])
                sim2w = embs.V[i].dot(embs.V[id2])
                sim3w = embs.V[i].dot(embs.V[id3])

                cosadd = sim2w + sim3w - sim1w 
                cosmul = sim2w * sim3w / sim1w

                if cosadd > best_add:
                    best_add = cosadd
                    best_w_add = embs.dictionary.get_token(i)

                if cosmul > best_mul:
                    best_mul = cosmul
                    best_w_mul = embs.dictionary.get_token(i)

            correct_cosadd += 1 if w4 == best_w_add else 0
            correct_cosmul += 1 if w4 == best_w_mul else 0

        # save the accuracies
        results[dname] = {
            '3cosadd': correct_cosadd / len(samples),
            '3cosmul': correct_cosmul / len(samples) }

    bar.finish()
    return results


def pos_tag_exp(embs, hdataset):
    pass


def chunking_exp(embs, hdataset):
    pass


def sentiment_exp(embs, hdataset):
    pass


def news_exp(embs, hdataset):
    pass


NAMES_TO_FUN = {
    'similarity': similarity_exp,
    'analogy': analogy_exp,
    'pos_brown': pos_tag_exp,
    'conll2000_chunking': chunking_exp,
    'imdb_sentiment': sentiment_exp,
    'agnews': news_exp,
}


# main function
def run_experiments(embs, datasets):
    all_results = {}
    for dname, hilbertd in datasets.items():
        exp = NAMES_TO_FUN[dname]
        all_results[dname] = exp(embs, hilbertd)
    return all_results


#### utility functions ###
def load_embeddings(path, vocab):
    if path == 'RANDOM':
        words = [t[0] for t in vocab if t[1] > 1] # don't get single occurences
        words_d = hilbert.dictionary.Dictionary(words)
        return hilbert.embeddings.random(d=EMB_DIM, vocab=len(words), dictionary=words_d, device='cpu')
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
    print('Loading datasets...')
    m_datasets = np.load('np/all_data.npz')['arr_0'][0]

    print('Loading/building embeddings...')
    m_vocab_tups = get_all_words(m_datasets.values())
    m_emb = load_embeddings(args.emb_path, m_vocab_tups)

    # run the experiments!
    exps = run_experiments(m_emb, m_datasets)
    for m_name, m_res in exps.items():
        if m_res is None: continue
        print('Results for {}:'.format(m_name))
        if type(m_res) == dict:
            for m_key, m_item in m_res.items():
                print('\t{}: {}'.format(m_key, m_item))


