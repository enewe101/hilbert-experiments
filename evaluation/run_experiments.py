import hilbert
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import spearmanr
from progress.bar import IncrementalBar
from collections import defaultdict
from dataset_load import HilbertDataset # required import to load numpy
from evaluation.train_classifier import train_classifier
from evaluation.train_seq_labeller import train_seq_labeller
from evaluation.torch_model import LogisticRegression, FFNN, SeqLabLSTM
from evaluation.constants import *
from evaluation.results import ResultsHolder
from evaluation.hparams import HParams


# little helper
def cossim(v1, v2):
    dot = v1.dot(v2)
    return dot / (v1.norm() * v2.norm())


# Beginning of our experimental code.
def similarity_exp(embs, hdataset, hparams):
    """
    Runs all 11 of the word similarity experiments on the set of
    embeddings passed to it.
    :param embs: Embeddings class, a hilbert embeddings object.
    :param hdataset: HilbertDataset object
    :param hparams: unused - kept for interface functionality
    :return: ResultsHolder object
    """
    results = ResultsHolder(SIMILARITY)

    # for showing over time
    print('Running similarity experiments')

    # iterate over all the similarity datasets in the object
    for dname, samples in hdataset.items():
        similarities = []
        gold = []
        had_coverage = []
        for w1, w2, gold_score in samples:
            had_coverage.append(w1 in embs.dictionary and w2 in embs.dictionary)
            gold.append(gold_score)

            e1 = embs.get_vec(w1, oov_policy='unk')
            e2 = embs.get_vec(w2, oov_policy='unk')
            similarities.append(cossim(e1, e2).item())

        covered = [(p, g) for i, (p, g) in enumerate(zip(similarities,gold)) if had_coverage[i]]
        results.add_ds_results(dname, {
            'full-spearman': spearmanr(gold, similarities)[0],
            'covered-spearman': spearmanr(*zip(*covered))[0],
            'coverage': sum(had_coverage) / len(had_coverage),
        })
    
    return results


def analogy_exp(embs, hdataset, hparams):
    """
    Runs the two big analogy datasets on the set of embeddings passed
    to it. Calculates 3cosadd and 3cosmul.
    :param embs: Embeddings class, a hilbert embeddings object.
    :param hdataset: HilbertDataset object
    :param hparams: unused - kept for interface functionality
    :return: ResultsHolder object
    """
    results = ResultsHolder(ANALOGY)

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

            # calculuate 3cosadd
            cos_add = sim_all[:,1] + sim_all[:,2] - sim_all[:,0]

            # 3cosmul requires all similarities to be nonnegative, conveniently told to us in a footnote.
            # see footnote 7 in http://anthology.aclweb.org/W/W14/W14-1618.pdf
            sim_all = (sim_all + 1) / 2
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
        results.add_ds_results(dname, {
            '3cosadd': correct_cosadd / len(samples),
            '3cosmul': correct_cosmul / len(samples),
            '3cosadd_had_answer': correct_cosadd / (len(samples) - missing_answer),
            '3cosmul_had_answer': correct_cosmul / (len(samples) - missing_answer),
            '3cosadd_full_coverage': correct_cosadd / total_all_embeddings,
            '3cosmul_full_coverage': correct_cosmul / total_all_embeddings,
            'missing_words': missing_words / (3 * len(samples)),
            'missing_answer': missing_answer / len(samples),
            'coverage': total_all_embeddings / len(samples),
        })

    bar.finish()
    return results


# TODO: write code to tune the hyperparams of ALL of the models.
# TODO: considering adding a CRF on top of the LSTM predictions.
# TODO: serialize results systematically.
# TODO: improve documentation.

def seq_labelling_exp(embs, hdataset, hparams):
    """

    :param embs:
    :param hdataset:
    :param hparams: HParams
    :return:
    """
   
    # get the training data
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True, translate_label_by_one=True)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True, translate_label_by_one=True)

    # x is list of sentences, sentence is list of tokens
    # y is list of pos-tag lists for each token
    neural_constructor = SeqLabLSTM
    neural_kwargs = {'n_labels': len(hdataset.labels_to_idx),
                     'rnn_hdim': hparams.rnn_hdim,
                     'n_layers': hparams.n_layers,
                     'dropout': hparams.dropout}

    results = train_seq_labeller(hdataset.name,
                                 embs,
                                 neural_constructor,
                                 neural_kwargs,
                                 lr=hparams.lr,
                                 n_epochs=250,
                                 mb_size=hparams.mb_size,
                                 early_stop=20,
                                 tr_x=tr_x,
                                 tr_y=tr_y,
                                 te_x=te_x,
                                 te_y=te_y,
                                 normalize_gradient=hparams.normalize_gradient,
                                 schedule_lr=hparams.schedule_lr,
                                 verbose=True,)
    return results


def classification_exp(embs, hdataset, hparams):
    """

    :param embs:
    :param hdataset:
    :param hparams: HParams
    :return:
    """
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True, translate_label_by_one=False)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True, translate_label_by_one=False)

    # set the neural constructor
    if hparams.model_str.lower() == 'ffnn':
        neural_constructor = FFNN
    elif hparams.model_str.lower() == 'logreg':
        neural_constructor = LogisticRegression
    else:
        raise NotImplementedError('Constructor model \"{}\" not '
                                  'implemented!'.format(hparams.model_str))

    neural_kwargs = {'n_classes': len(hdataset.labels_to_idx)}

    # special parameters for a FFNN
    if hparams.model_str == 'ffnn':
        neural_kwargs.update({'hdim1': hparams.hdim1,
                              'hdim2': hparams.hdim2,
                              'dropout': hparams.dropout})

    results = train_classifier(hdataset.name,
                               embs,
                               neural_constructor,
                               neural_kwargs,
                               lr=hparams.lr,
                               n_epochs=250,
                               mb_size=hparams.mb_size,
                               early_stop=20,
                               tr_x=tr_x,
                               tr_y=tr_y,
                               te_x=te_x,
                               te_y=te_y,
                               schedule_lr=hparams.schedule_lr,
                               verbose=True,)
    return results


#### utility functions ###
def load_embeddings(path, device=None):
    e = hilbert.embeddings.Embeddings.load(path,
            device=HParams.DEVICE.type if device is None else device)
    if len(e.V) == EMB_DIM:
        e.V = e.V.transpose(0, 1)
    return e


def get_all_words(list_of_hdatasets):
    vocab = {}
    for d in list_of_hdatasets:
        vocab.update(d.get_unique_words_counts())
    return list(sorted(vocab.items(), key=lambda t: t[1]))



if __name__ == '__main__':
    ## Adding m_ to all variable names to indicate they only belong in main, not globals.
    m_hparams = HParams() # parses args

    # load up the datasets and get the vocab we will need.
    print('Loading datasets...')
    m_datasets = np.load('np/all_data.npz')['arr_0'][0]

    print('Loading embeddings...')
    m_emb = load_embeddings(m_hparams.emb_path)

    m_names_to_fun = {
        SIMILARITY: similarity_exp,
        ANALOGY: analogy_exp,
        BROWN_POS: seq_labelling_exp,
        WSJ_POS: seq_labelling_exp,
        CHUNKING: seq_labelling_exp,
        SENTIMENT: classification_exp,
        NEWS: classification_exp,
    }

    m_exp = m_names_to_fun[m_hparams.experiment]
    m_ds = m_datasets[m_hparams.experiment]
    m_results = m_exp(m_emb, m_ds, m_hparams)

    # TODO: formalize results serialization
    m_results.pretty_print()
