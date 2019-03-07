import hilbert
import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.stats import spearmanr
from progress.bar import IncrementalBar
from dataset_load import HilbertDataset # required import to load numpy
from evaluation.train_classifier import train_classifier
from evaluation.train_seq_labeller import train_seq_labeller
from evaluation.torch_model import LogisticRegression, FFNN, SeqLabLSTM, BiLSTMClassifier
from evaluation.constants import *
from evaluation.results import ResultsHolder
from evaluation.hparams import HParams


# to help with dealing with averaging vectors and covectors
class EmbWrapper(object):
    def __init__(self, hembs, avg_vw=False):
        self.dictionary = hembs.dictionary
        self.matrix = hembs.V
        self.dim = len(self.matrix[0])
        self.unk = hembs.unk
        if avg_vw:
            if hembs.W is not None:
                self.matrix += hembs.W
                self.matrix /= 2.
            else:
                print('(weak warning) no covectors found!')

    def get_id(self, w):
        return self.dictionary.get_id(w)

    def get_emb(self, w):
        try:
            idx = self.dictionary.get_id(w)
            return self.matrix[idx]
        except KeyError:
            return self.unk

    def get_token(self, idx):
        return self.dictionary.get_token(idx)

    def has_w(self, w):
        return w in self.dictionary

    def has_nan(self):
        return any(np.isnan(self.matrix[0]))



# little helper
def cossim(v1, v2):
    dot = v1.dot(v2)
    return dot / (v1.norm() * v2.norm())


# Beginning of our experimental code.
def similarity_exp(embs, hdataset, hparams, verbose=True):
    """
    Runs all 11 of the word similarity experiments on the set of
    embeddings passed to it.
    :param embs: EmbWrapper class, a wrapper for hilbert embeddings object.
    :param hdataset: HilbertDataset object
    :param hparams: unused - kept for interface functionality
    :param verbose: print or not
    :return: ResultsHolder object
    """
    results = ResultsHolder(SIMILARITY)

    # for showing over time
    if verbose:
        print('Running similarity experiments')

    # iterate over all the similarity datasets in the object
    for dname, samples in hdataset.items():
        if dname == 'semeval17task2_trial':
            continue # don't do this shitty one as it only has 18 samples!

        similarities = []
        gold = []
        had_coverage = []
        for w1, w2, gold_score in samples:
            had_coverage.append(embs.has_w(w1) and embs.has_w(w2))
            gold.append(gold_score)

            e1 = embs.get_emb(w1)
            e2 = embs.get_emb(w2)
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
    embs.matrix = F.normalize(embs.matrix, p=2, dim=1)
    
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

            if not embs.has_w(w4):
                missing_answer += 1
                continue

            e1 = embs.get_emb(w1).reshape(-1, 1)
            e2 = embs.get_emb(w2).reshape(-1, 1)
            e3 = embs.get_emb(w3).reshape(-1, 1)

            # get cos sims for each of them with the dataset
            sim_all = embs.matrix.mm(torch.cat([e1, e2, e3], dim=1))

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
                    w_id = embs.get_id(wi)
                    cos_add[w_id] = -np.inf
                    cos_mul[w_id] = -np.inf
                except KeyError:
                    missing_words += 1
                    have_all_embs = False

            # get the best with argmax
            best_w_add = embs.get_token(cos_add.argmax())
            best_w_mul = embs.get_token(cos_mul.argmax())

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


# TODO: add CRF on top of the LSTM predictions.
# TODO: improve documentation.

def seq_labelling_exp(embs, hdataset, hparams):
    """
    Run a sequence labelling experiment over some dataset. For example,
    over the wsj-pos or brown-pos datasets/Hilbert experiments. Returns
    a nice results object with everything stored in it.
    :param embs: HilbertEmbeddings object
    :param hdataset: HilbertDataset object (for sequence labelling)
    :param hparams: HParams
    :return: ResultsHolder
    """

    # get the training data
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True, translate_label_by_one=True)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True, translate_label_by_one=True)

    # deal with supersense-specific label evaluation (still have 0 label for output layer though!)
    sst_labels = None
    if 'sst' in hdataset.name:
        sst_labels = set(hdataset.labels_to_idx.values())
        sst_labels.remove(hdataset.ignore_idx)
        assert hdataset.ignore_idx > 0
        sst_labels = sorted(list(sst_labels))
    
    # x is list of sentences, sentence is list of tokens
    # y is list of pos-tag lists for each token
    neural_constructor = SeqLabLSTM
    neural_kwargs = {'n_labels': len(hdataset.labels_to_idx),
                     'rnn_hdim': hparams.rnn_hdim,
                     'n_layers': hparams.n_layers,
                     'dropout': hparams.dropout,
                     'fine_tune': hparams.fine_tune}

    results = train_seq_labeller(hdataset.name,
                                 embs,
                                 neural_constructor,
                                 neural_kwargs,
                                 lr=hparams.lr,
                                 n_epochs=hparams.epochs,
                                 mb_size=hparams.mb_size,
                                 early_stop=10,
                                 tr_x=tr_x,
                                 tr_y=tr_y,
                                 te_x=te_x,
                                 te_y=te_y,
                                 normalize_gradient=hparams.normalize_gradient,
                                 schedule_lr=hparams.schedule_lr,
                                 sst_labels=sst_labels,
                                 verbose=True,)
    return results


def classification_exp(embs, hdataset, hparams):
    """
    Run a classification experiment over some dataset. For example,
    over the sentiment or news datasets. Returns results object.
    :param embs: HilbertEmbeddings object
    :param hdataset: HilbertDataset object (for classification)
    :param hparams: HParams
    :return: ResultsHolder
    """
    tr_x, tr_y = hdataset.get_x_y('train', embs.dictionary, as_indicies=True, translate_label_by_one=False)
    te_x, te_y = hdataset.get_x_y('test', embs.dictionary, as_indicies=True, translate_label_by_one=False)

    # set the neural constructor
    if hparams.model_str.lower() == 'ffnn':
        neural_constructor = FFNN
    elif hparams.model_str.lower() == 'logreg':
        neural_constructor = LogisticRegression
    elif hparams.model_str.lower() == 'bilstm':
        neural_constructor = BiLSTMClassifier
    else:
        raise NotImplementedError('Constructor model \"{}\" not '
                                  'implemented!'.format(hparams.model_str))

    neural_kwargs = {'n_classes': len(hdataset.labels_to_idx),
                     'fine_tune': hparams.fine_tune}

    # special parameters for a FFNN
    if hparams.model_str.lower() == 'ffnn':
        neural_kwargs.update({'hdim1': hparams.hdim1,
                              'hdim2': hparams.hdim2,
                              'dropout': hparams.dropout})

    elif hparams.model_str.lower() == 'bilstm':
        neural_kwargs.update({'rnn_hdim': hparams.rnn_hdim,
                              'n_layers': hparams.n_layers,
                              'dropout': hparams.dropout})

    # run the model!
    exp_name = '{}_{}'.format(hdataset.name, hparams.model_str.lower())
    results = train_classifier(exp_name,
                               embs,
                               neural_constructor,
                               neural_kwargs,
                               lr=hparams.lr,
                               n_epochs=hparams.epochs,
                               mb_size=hparams.mb_size,
                               early_stop=10,
                               tr_x=tr_x,
                               tr_y=tr_y,
                               te_x=te_x,
                               te_y=te_y,
                               schedule_lr=hparams.schedule_lr,
                               verbose=True,)
    return results


#### utility functions ###
def load_embeddings(path, device=None, avg_vw=False):
    e = hilbert.embeddings.Embeddings.load(path,
            device=HParams.DEVICE.type if device is None else device)
    if len(e.V) == EMB_DIM:
        e.V = e.V.transpose(0, 1)
    return EmbWrapper(e, avg_vw)


def get_all_words(list_of_hdatasets):
    vocab = {}
    for d in list_of_hdatasets:
        vocab.update(d.get_unique_words_counts())
    return list(sorted(vocab.items(), key=lambda t: t[1]))


def main():
    hparams = HParams() # parses args

    # load up the datasets and get the vocab we will need.
    print('Loading datasets...')
    datasets = np.load(hparams.data_path)['arr_0'][0]

    for emb_path in hparams.iter_emb_paths():
        print('Loading embeddings from {}..'.format(emb_path))
        emb = load_embeddings(emb_path, avg_vw=hparams.avgvw)
        if hparams.avgvw:
            print('-- averaging vectors and covectors --')

        names_to_fun = {
            SIMILARITY: similarity_exp,
            BROWN_POS: seq_labelling_exp,
            WSJ_POS: seq_labelling_exp,
            SST: seq_labelling_exp,
            # CHUNKING: seq_labelling_exp,
            SENTIMENT: classification_exp,
            NEWS: classification_exp,
            ANALOGY: analogy_exp,
        }

        model_str = ''
        if hparams.experiment in (NEWS, SENTIMENT):
            model_str = '_' + hparams.model_str

        for expname, exp in names_to_fun.items():
            if hparams.experiment != 'all' and expname != hparams.experiment:
                continue

            # we may be running repeated experiments
            mean_results = ResultsHolder(expname + model_str)
            params_str = hparams.get_params_str()
            for i in range(hparams.repeat):
                
                # flush the GPU and wait a second to reset everything
                torch.cuda.empty_cache()
                time.sleep(1)

                # set the seed right before running experiment
                hparams.seed += (i * 1917)
                np.random.seed(hparams.seed)
                torch.random.manual_seed(hparams.seed)

                # run it and get the results!
                results = exp(emb, datasets[expname], hparams)

                # save temp results
                results.serialize(os.path.join(emb_path, 'tmpres/'), params_str)

                if hparams.repeat == 1:
                    mean_results = results
                else:
                    got_res = {**results.results_by_dataset['full']}
                    mean_results.add_ds_results(
                        'seed {}'.format(hparams.seed), got_res)

            mean_results.serialize(emb_path, params_str)




if __name__ == '__main__':
    main()
