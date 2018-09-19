import argparse
import hilbert
import numpy as np
import sklearn_crfsuite as skcrf
from sklearn_crfsuite import metrics
from nltk.corpus import stopwords
from dataset_load import HilbertDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.stats import spearmanr
from progress.bar import IncrementalBar


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


# feature extraction function for CRFs
# need feature extraction function to apply to each sentence
# this dictionary based method is somewhat bad, but it is the way their API works
def _crf_features(idx, tok_embs, only_emb=True):
    # start with basic features
    f = {} if only_emb else {
        'is_first': idx == 0,
        'is_last': idx == len(tok_embs) - 1,
    } 
    
    # now we add word embeddings as features
    emb = tok_embs[idx]
    f.update({'e{}'.format(i): emb[i] 
              for i in range(len(emb))})
    
    if only_emb:
        return f
    
    # add previous and last embeddings as features, if applicable
    surr_embs = []
    if idx > 0:
        surr_embs.append(('prev_e', tok_embs[idx-1]))
    if idx < len(tok_embs) - 1:
        surr_embs.append(('post_e', tok_embs[idx+1]))
    
    for prefix, e in surr_embs:
        f.update({'{}{}'.format(prefix, i): e[i] 
                  for i in range(len(e))})
    
    return f



def pos_tag_exp(embs, hdataset, safe=True):
   
    # get the training data
    tr_x, tr_y = hdataset.get_x_y('train')
    te_x, te_y = hdataset.get_x_y('test')

    # x is list of sentences, sentence is list of tokens
    # y is list of pos-tag lists for each token

    # now we iterate over everything and extract features
    bar = IncrementalBar('extracting features', max=len(tr_x)+len(te_x))
    X = []
    for sent_tokens in tr_x + te_x:
        sent_embeddings = [embs.get_vec(t, oov_policy='unk') 
                           for t in sent_tokens]
        X.append([_crf_features(idx, sent_embeddings)
                  for idx in range(len(sent_embeddings))])
        bar.next()
    bar.finish()

    # set the train and test sets with slicing
    X_train = X[:len(tr_x)]
    X_test = X[len(tr_x):]
    
    # basic safety checking
    if safe:
        for _x, _y in [(X_train, tr_y), (X_test, te_y)]:
            assert len(_x) == len(_y)
            for i, sent_feats in enumerate(_x):
                assert len(sent_feats) == len(_y[i])

    # now build the model
    print('Fitting crf...')
    model = skcrf.CRF(verbose=True)
    model.fit(X_train, tr_y)
    y_pred = model.predict(X_test)

    # accuracy
    acc = metrics.flat_accuracy_score(te_y, y_pred)
    return {'flat_acc': acc}


def chunking_exp(embs, hdataset):
    pass


def sentiment_exp(embs, hdataset):
    tr_x, tr_y = hdataset.get_x_y('train')
    te_x, te_y = hdataset.get_x_y('test')
    X_feats = []
    
    # make features be the mean embedding vector, simple.
    bar = IncrementalBar('Extracting features', max=len(tr_x) + len(te_x))
    for sample in tr_x + te_x:
        features = np.zeros(EMB_DIM)
        count = 0
        for token in filter(lambda t: t not in SENTI_STOPS, sample):
            try:
                features += embs[token]
                count += 1
            except KeyError:
                continue
        features = embs.unk if count == 0 else features / count
        X_feats.append(features)
        bar.next()
    bar.finish()

    # train log reg on the samples
    print('Fitting model...')
    X_feats = np.array(X_feats)
    model = LogisticRegression()
    model.fit(X_feats[ :len(tr_x)], tr_y)

    # predict, return results dict
    y_pred = model.predict( X_feats[len(tr_x): ] )
    return {
        'accuracy': accuracy_score(te_y, y_pred),
        'report': classification_report(te_y, y_pred),
    }


def news_exp(embs, hdataset):
    tr_x, tr_y = hdataset.get_x_y('train')
    te_x, te_y = hdataset.get_x_y('test')
    X_feats = []
    
    # make features be the mean embedding vector, simple.
    bar = IncrementalBar('Extracting features', max=len(tr_x) + len(te_x))
    for sample in tr_x + te_x:
        features = np.zeros(EMB_DIM)
        count = 0
        for token in filter(lambda t: t not in stopwords, sample):
            try:
                features += embs[token]
                count += 1
            except KeyError:
                continue
        features = embs.unk if count == 0 else features / count
        X_feats.append(features)
        bar.next()
    bar.finish()

    # train log reg on the samples
    print('Fitting model...')
    X_feats = np.array(X_feats) 
    model = LogisticRegression()
    model.fit(X_feats[ :len(tr_x)], tr_y)

    # predict, return results dict
    y_pred = model.predict( X_feats[len(tr_x): ] )
    return {
        'accuracy': accuracy_score(te_y, y_pred),
        'report': classification_report(te_y, y_pred),
    }


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
def run_experiments(embs, datasets, option='all'):
    all_results = {}
    for dname, hilbertd in datasets.items():
        if option == 'all' or dname == option:
            exp = NAMES_TO_FUN[dname]
            all_results[dname] = exp(embs, hilbertd)
            print(all_results[dname])
    return all_results


#### utility functions ###
def load_embeddings(path, vocab):
    if path == 'RANDOM':
        words = [t[0] for t in vocab if t[1] > 1] # don't get single occurences
        words_d = hilbert.dictionary.Dictionary(words)
        return hilbert.embeddings.random(d=EMB_DIM, 
                distribution='uniform',
                scale=0.5,
                vocab=len(words), 
                dictionary=words_d, 
                device='cpu',
        )
    else:
        e = hilbert.embeddings.Embeddings.load(
            path,
            device='cpu',
        )
        if len(e.V) == 300:
            e.V = e.V.transpose(0, 1)
        e.V = e.V.numpy()
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
    parser.add_argument('--emb_path', type=str, default='RANDOM',
        help='path to the embeddings we want to process,'
             'default is just using a constant random init.'
    )
    parser.add_argument('--exp', type=str, default='all',
        choices=['all', 'sentiment', 'pos', 'similarity', 
                 'analogy', 'chunking', 'news'],
        help='specific experiment to run, use for debugging'
    )
    args = parser.parse_args() 

    # load up the datasets and get the vocab we will need.
    print('Loading datasets...')
    m_datasets = np.load('np/all_data.npz')['arr_0'][0]

    print('Loading/building embeddings...')
    m_vocab_tups = get_all_words(m_datasets.values())
    m_emb = load_embeddings(args.emb_path, m_vocab_tups)

    # run the experiments!
    exps = run_experiments(m_emb, m_datasets, option=args.exp)
    for m_name, m_res in exps.items():
        if m_res is None: continue
        print('Results for {}:'.format(m_name))
        if type(m_res) == dict:
            for m_key, m_item in m_res.items():
                print('\t{}: {}'.format(m_key, m_item))


