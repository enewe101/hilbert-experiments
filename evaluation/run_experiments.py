import argparse
import hilbert
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from scipy.stats import spearmanr
from progress.bar import IncrementalBar
from dataset_load import HilbertDataset
from evaluation.train_classifier import train_classifier
from evaluation.torch_model import LogisticRegression, FFNN
from hilbert_device import DEVICE


# little helper
def cossim(v1, v2):
    dot = v1.dot(v2)
    return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))


# global constants
stopwords = set(stopwords.words('english'))
SENTI_STOPS = stopwords.copy()
SENTI_STOPS.difference_update({'no', 'not'})
SENTI_STOPS.update({'.'})
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
            e1 = embs.get_vec(w1, oov_policy='unk')
            e2 = embs.get_vec(w2, oov_policy='unk')
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
            e1 = embs.get_vec(w1, oov_policy='unk')
            e2 = embs.get_vec(w2, oov_policy='unk')
            e3 = embs.get_vec(w3, oov_policy='unk')

            # get cos sims for each of them with the dataset
            ## TODO: combine e1, e2, e3 together and do one mm (maybe it would be slightly faster?)
            sim1_all = embs.V.mm(e1.reshape(-1, 1))
            sim2_all = embs.V.mm(e2.reshape(-1, 1))
            sim3_all = embs.V.mm(e3.reshape(-1, 1))
            cos_add = sim2_all + sim3_all - sim1_all
            cos_mul = sim2_all * sim3_all / (sim1_all + 0.0001) # add epsilon to avoid divide by 0

            # make sure we don't get the vecs themselves
            for wi in (w1, w2, w3):
                cos_add[embs.dictionary.get_id(wi)] = -np.inf
                cos_mul[embs.dictionary.get_id(wi)] = -np.inf

            # get the best with argmax
            best_w_add = embs.dictionary.get_token(cos_add.argmax())
            best_w_mul = embs.dictionary.get_token(cos_mul.argmax())

            # count up for final accuracy
            correct_cosadd += 1 if w4 == best_w_add else 0
            correct_cosmul += 1 if w4 == best_w_mul else 0

        # save the accuracies
        results[dname] = {
            '3cosadd': correct_cosadd / len(samples),
            '3cosmul': correct_cosmul / len(samples), 
        }

    bar.finish()
    return results


def pos_tag_exp(embs, hdataset, safe=True):
   
    # get the training data
    tr_x, tr_y = hdataset.get_x_y('train')
    te_x, te_y = hdataset.get_x_y('test')

    # x is list of sentences, sentence is list of tokens
    # y is list of pos-tag lists for each token


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
                               lr=0.001,
                               n_epochs=50,
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
        neural_kwargs.update({'hdim1': 128, 'hdim2': 128})
    results = train_classifier(embs,
                               neural_constructor,
                               neural_kwargs,
                               lr=0.001,
                               n_epochs=50,
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
        choices=['all', 'sentiment', 'pos', 'similarity', 
                 'analogy', 'chunking', 'news'],
        help='specific experiment to run, use for debugging'
    )
    parser.add_argument('-c', '--classifier', type=str, default='logreg',
        help='classifier to use in the classification experiments'
    )
    args = parser.parse_args() 

    # load up the datasets and get the vocab we will need.
    print('Loading datasets...')
    m_datasets = np.load('np/all_data.npz')['arr_0'][0]

    print('Loading/building embeddings...')
    m_emb = load_embeddings(args.emb_path)

    m_kwargs = {'news': {'torch_model_str': args.classifier},
                'sentiment': {'torch_model_str': args.classifier}}

    # run the experiments!
    exps = run_experiments(m_emb, m_datasets, kwargs=m_kwargs, option=args.exp)
    for m_name, m_res in exps.items():
        if m_res is None: continue
        print('Results for {}:'.format(m_name))
        if type(m_res) == dict:
            for m_key, m_item in m_res.items():
                print('\t{}: {}'.format(m_key, m_item))


