import os
import sys

import numpy as np
import torch

import shared
import hilbert as h
import hilbert_noshard as hn
import data_preparation as dp
import embedding_creation as ec


K = 15
LEARNING_RATE = 1e-7
EMBEDDINGS_INIT_PATH = 'w2v-testing'
DEVICE = 'cpu'
DTYPE = torch.float32
ALPHA = 3./4.
T = 1e-5


def get_w2v(replicate):
    """
    Basic validation of equivalence of Hilbert embedder with word2vec.
    """
    np.random.seed(replicate)
    torch.random.manual_seed(replicate)
    cooc_stats_path = '5w-dynamic-10k/5w-dynamic-10000-rep{}'.format(replicate)
    cooc_stats = dp.read_stats(cooc_stats_path)
    embedder = h.embedder.get_w2v_embedder(
        cooc_stats=cooc_stats,
        k=K,
        alpha=ALPHA,
        t=T,
        learning_rate=LEARNING_RATE,
        shard_factor=1
    )
    return embedder, cooc_stats


def run_w2v_replicate(replicate):
    print('replicate\t{}'.format(replicate))
    iters_per_period = 10000
    num_periods = 50

    dirname = 'hbt-w2v-10k-rep{}'.format(replicate)
    dirpath = os.path.join(shared.CONSTANTS.EMBEDDINGS_DIR, dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    embedder, cooc_stats = get_w2v(replicate)
    describe(embedder)
    for period in range(num_periods):
        print('period\t{}'.format(period))
        embedder.cycle(times=iters_per_period)
        embeddings = h.embeddings.Embeddings(
            V=embedder.V, W=embedder.W, dictionary=cooc_stats.dictionary)
        iters_thousands = round(iters_per_period * period / 1000.)
        fname = 'hbt-w2v-10k-rep{}-iter{}k'.format(replicate, iters_thousands)
        embeddings.save(os.path.join(dirpath, fname))

    return embedder


def describe(embedder):
    print('> embedder')
    print('f_delta\t{}'.format(embedder.delta.__class__.__name__))
    print('d\t{}'.format(embedder.d))
    print('vocab\t{}'.format(embedder.num_vecs))
    print('learning_rate\t{}'.format(embedder.learning_rate))
    print('one_sided\t{}'.format(embedder.one_sided))
    constrainer_name = (
        'None' if not embedder.constrainer else embedder.constrainer.__name__)
    print('constrainer\t{}'.format(constrainer_name))
    print('shard_factor\t{}'.format(embedder.shard_factor))
    print('device\t{}'.format(embedder.device))

    f_delta = embedder.delta
    print('> f_delta')
    print('k\t{}'.format(f_delta.k))

    M = f_delta.M
    print('> M')
    print('base\t{}'.format(M.base.__name__))
    print('shift_by\t{}'.format(M.shift_by))
    print('neg_inf_val\t{}'.format(M.neg_inf_val))
    print('clip_thresh\t{}'.format(M.clip_thresh))
    print('diag\t{}'.format(M.diag))
    print('base_args\t{}'.format(str(M.base_args)))




if __name__ == '__main__':
    replicate = int(sys.argv[1])
    run_w2v_replicate(replicate)

