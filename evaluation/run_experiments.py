import argparse
import os
import hilbert
import rmsd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from scipy.stats import spearmanr
from progress.bar import IncrementalBar
from collections import defaultdict
from dataset_load import HilbertDataset # required import to load numpy
from evaluation.train_classifier import train_classifier
from evaluation.train_seq_labeller import train_seq_labeller
from evaluation.torch_model import LogisticRegression, FFNN, SeqLabLSTM
from hilbert_device import DEVICE


# little helper
def cossim(v1, v2):
    dot = v1.dot(v2)
    return dot / (v1.norm() * v2.norm())


# global constants
stopwords = set(stopwords.words('english'))
SENTI_STOPS = stopwords.copy()
SENTI_STOPS.difference_update({'no', 'not'})
SENTI_STOPS.update({'.'})
EMB_DIM = 300


def similarity_exp(embs, hdataset):
    results = {}

    # for showing over time
    print('Running similarity experiments')

    # iterate over all the similarity datasets in the object
    for dname, samples in hdataset.items():
        similarities = []
        gold = []
        for w1, w2, gold_score in samples:
            gold.append(gold_score)

            e1 = embs.get_vec(w1, oov_policy='unk')
            e2 = embs.get_vec(w2, oov_policy='unk')
            similarities.append(cossim(e1, e2).item())
        
        results[dname] = spearmanr(gold, similarities)[0]
    
    return results


def analogy_exp(embs, hdataset):
    results = {}

    # normalize for faster sim calcs.
    embs.V = F.normalize(embs.V, p=2, dim=1)
    
    # for showing over time
    total_iter = sum(len(samples) for samples in hdataset.values())
    iter_step = 100
    bar = IncrementalBar('Running analogy experiments', max=total_iter)
   
    # iterate over the two analogy datasets
    for dname, samples in hdataset.items():
        correct_cosadd = 0
        correct_cosmul = 0
        missing_words = 0
        missing_answer = 0
        total_all_embeddings = 0

        # w1 is to w2 as w3 is to w4
        for i, (w1, w2, w3, w4) in enumerate(samples):
            if i % iter_step == 0: bar.next(n=iter_step)

            if not w4 in embs.dictionary:
                missing_answer += 1
                continue

            e1 = embs.get_vec(w1, oov_policy='unk').reshape(-1, 1)
            e2 = embs.get_vec(w2, oov_policy='unk').reshape(-1, 1)
            e3 = embs.get_vec(w3, oov_policy='unk').reshape(-1, 1)

            # get cos sims for each of them with the dataset
            sim_all = embs.V.mm(torch.cat([e1, e2, e3], dim=1))
            #sim1_all, sim2_all, sim3_all = sim_all[:,0], sim_all[:,1], sim_all[:,2]
            cos_add = sim_all[:,1] + sim_all[:,2] - sim_all[:,0]
            cos_mul = (sim_all[:,1] * sim_all[:,2]) / (sim_all[:,0] + 0.001) # add epsilon to avoid divide by 0

            # make sure we don't get the vecs themselves
            have_all_embs = True
            for wi in (w1, w2, w3):
                try:
                    w_id = embs.dictionary.get_id(wi)
                    cos_add[w_id] = -np.inf
                    cos_mul[w_id] = -np.inf
                except KeyError:
                    missing_words += 1
                    have_all_embs = False

            # get the best with argmax
            best_w_add = embs.dictionary.get_token(cos_add.argmax())
            best_w_mul = embs.dictionary.get_token(cos_mul.argmax())

            # count up for final accuracy
            correct_cosadd += 1 if w4 == best_w_add else 0
            correct_cosmul += 1 if w4 == best_w_mul else 0
            total_all_embeddings += 1 if have_all_embs else 0

        # save the accuracies
        results[dname] = {
            '3cosadd': correct_cosadd / len(samples),
            '3cosmul': correct_cosmul / len(samples),
            '3cosadd_hadanswer': correct_cosadd / (len(samples) - missing_answer),
            '3cosmul_hadanswer': correct_cosmul / (len(samples) - missing_answer),
            '3cosadd_fullcoverage': correct_cosadd / total_all_embeddings,
            '3cosmul_fullcoverage': correct_cosmul / total_all_embeddings,
            'missing_words': missing_words / (3 * len(samples)),
            'coverage': total_all_embeddings / len(samples),
            'missing_answer': missing_answer / len(samples),
        }

    bar.finish()
    return results


def pos_tag_exp(embs, hdataset):
   
    # get the training data
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True)

    # x is list of sentences, sentence is list of tokens
    # y is list of pos-tag lists for each token
    neural_constructor = SeqLabLSTM
    neural_kwargs = {'n_labels': len(hdataset.labels_to_idx),
                     'hdim': 256,
                     'n_layers': 2}

    results = train_seq_labeller(embs,
                                 neural_constructor,
                                 neural_kwargs,
                                 lr=0.001,
                                 n_epochs=250,
                                 mb_size=16,
                                 early_stop=15,
                                 tr_x=tr_x,
                                 tr_y=tr_y,
                                 te_x=te_x,
                                 te_y=te_y,
                                 verbose=True,)
    return results


def chunking_exp(embs, hdataset):
    pass


def sentiment_exp(embs, hdataset, torch_model_str):
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True)

    neural_constructor = FFNN if torch_model_str == 'ffnn' else LogisticRegression
    neural_kwargs = {'n_classes': len(hdataset.labels_to_idx)}
    if torch_model_str == 'ffnn':
        neural_kwargs.update({'hdim1': 128, 'hdim2': 128})
    results = train_classifier(embs,
                               neural_constructor,
                               neural_kwargs,
                               lr=0.0005,
                               n_epochs=150,
                               mb_size=64,
                               early_stop=10,
                               tr_x=tr_x,
                               tr_y=tr_y,
                               te_x=te_x,
                               te_y=te_y,
                               verbose=True,)
    return results


def news_exp(embs, hdataset, torch_model_str):
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True)

    neural_constructor = FFNN if torch_model_str == 'ffnn' else LogisticRegression
    neural_kwargs = {'n_classes': len(hdataset.labels_to_idx)}
    if torch_model_str == 'ffnn':
        neural_kwargs.update({'hdim1': 128, 'hdim2': 128, 'dropout': 0})
    results = train_classifier(embs,
                               neural_constructor,
                               neural_kwargs,
                               lr=0.0005,
                               n_epochs=150,
                               mb_size=64,
                               early_stop=10,
                               tr_x=tr_x,
                               tr_y=tr_y,
                               te_x=te_x,
                               te_y=te_y,
                               verbose=True,)
    return results


### primary running code below ###
NAMES_TO_FUN = {
    'similarity': similarity_exp,
    'analogy': analogy_exp,
    'pos': pos_tag_exp,
    'chunking': chunking_exp,
    'sentiment': sentiment_exp,
    'news': news_exp,
}


# main function
def run_experiments(embs, datasets, kwargs, option='all'):
    all_results = {}
    for dname, hilbertd in datasets.items():
        if option == 'all' or dname == option:
            exp = NAMES_TO_FUN[dname]
            all_results[dname] = exp(embs, hilbertd, **kwargs[dname])
            print(all_results[dname])
    return all_results


# compare the big boys
def run_embedding_comparisons(all_embs_dict):
    names = list(sorted(all_embs_dict.keys()))
    n = len(names) # I hate repeating len all the time

    # we will be making a triu matrix of all the rotation errors
    intrinsic_res = np.zeros((n, n))
    rand_res = np.zeros((n, n))
    n_intrinsic_res = np.zeros((n, n)) # normalized
    n_rand_res = np.zeros((n, n)) # normalized

    # send everything to CPU
    for key, hilbert_emb in all_embs_dict.items():
        all_embs_dict[key] = hilbert_emb.V.cpu().numpy()

    # iterate over the boys
    bar = IncrementalBar('Doing Kabsch algorithm...', max=n * n)
    for i, e1_name in enumerate(names):
        e1_V = all_embs_dict[e1_name]
        e1_norms = np.linalg.norm(e1_V, axis=1).reshape(-1, 1)

        for j in range(n):
            e2_V = all_embs_dict[names[j]] # allow compare with self for i==j

            # we will also be comparing to a normally distributed random matrix
            # with the same mean and scaling as those of the other embeddings.
            rand_V = np.random.normal(loc=e2_V.mean(),
                                      scale=e2_V.std(),
                                      size=e2_V.shape)
            rand_norms = np.linalg.norm(rand_V, axis=1).reshape(-1, 1)

            # sanity check
            assert e1_V.shape == e2_V.shape == rand_V.shape

            # we will be doing comparison with the randomly distributed vecs
            # in each scenario in order to have robust results
            rand_res[i, j] = rmsd.kabsch_rmsd(e1_V, rand_V)
            n_rand_res[i, j] = rmsd.kabsch_rmsd(e1_V / e1_norms, rand_V / rand_norms)

            # compare with the other vecs if j > i (otherwise we already did
            # that computation earlier for when i > j previously in loop)
            if j > i:
                e2_norms = np.linalg.norm(e2_V, axis=1).reshape(-1, 1)
                intrinsic_res[i, j] = rmsd.kabsch_rmsd(e1_V, e2_V)
                n_intrinsic_res[i, j] = rmsd.kabsch_rmsd(e1_V / e1_norms, e2_V / e2_norms)

            bar.next()
    bar.finish()

    np.set_printoptions(precision=4, suppress=True)
    print('Compared with each other:')
    print(' '.join(names))
    print(intrinsic_res)

    print('\nCompared to random:')
    print(' '.join(names))
    print(rand_res)

    print('\n[Normalized] Compared with each other:')
    print(' '.join(names))
    print(n_intrinsic_res)

    print('\n[Normalized] Compared to random:')
    print(' '.join(names))
    print(n_rand_res)



#### utility functions ###
def load_embeddings(path):
    e = hilbert.embeddings.Embeddings.load(
        path,
        device=DEVICE.type,
    )
    if len(e.V) == 300:
        e.V = e.V.transpose(0, 1)
    return e


def get_all_words(list_of_hdatasets):
    vocab = {}
    for d in list_of_hdatasets:
        vocab.update(d.get_unique_words_counts())
    return list(sorted(vocab.items(), key=lambda t: t[1]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run experiments on set of embeddings.'
    )
    parser.add_argument('emb_path', type=str,
        help='path to the embeddings we want to process,'
    )
    parser.add_argument('-e', '--exp', type=str, default='all',
        choices=['all', 'sentiment', 'pos', 'similarity', 'analogy',
                 'chunking', 'news'],
        help='specific experiment to run, use for debugging'
    )
    parser.add_argument('-c', '--classifier', type=str, default='logreg',
        help='classifier to use in the classification experiments'
    )
    parser.add_argument('--compare', action='store_true',
        help='if active, we will run the vector comparison protocol over '
             'all of the embeddings matching the pattern in emb_path arg;'
             ' i.e., rotation matrix fitting and average error.'
    )
    args = parser.parse_args()

    if args.compare:
        print('Loading all embeddings starting with pattern: {}...'.format(args.emb_path))
        all_embs = {}
        splitted = args.emb_path.split('/')
        directory = '/'.join(splitted[:-1])
        pattern = splitted[-1]
        for emb_dname in os.listdir(directory):
            if emb_dname.startswith(pattern):
                all_embs[emb_dname] = load_embeddings('{}/{}'.format(directory, emb_dname))
        run_embedding_comparisons(all_embs)
        exit(0)

    # load up the datasets and get the vocab we will need.
    print('Loading datasets...')
    m_datasets = np.load('np/all_data.npz')['arr_0'][0]

    print('Loading/building embeddings...')
    m_emb = load_embeddings(args.emb_path)

    m_kwargs = defaultdict(lambda: {})
    m_kwargs.update({'news': {'torch_model_str': args.classifier},
                     'sentiment': {'torch_model_str': args.classifier}})

    # run the experiments!
    exps = run_experiments(m_emb, m_datasets, kwargs=m_kwargs, option=args.exp)
    for m_name, m_res in exps.items():
        if m_res is None: continue
        print('Results for {}:'.format(m_name))
        if type(m_res) == dict:
            for m_key, m_item in m_res.items():
                try:
                    print('\t{:25}: {:.3f}'.format(m_key, m_item))
                except Exception:
                    print('\t{:25}: {}'.format(m_key, m_item))

