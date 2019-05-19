import argparse
import numpy as np
import os
import torch
#from evaluation.hparams import HParams
#from evaluation.run_experiments import load_embeddings
from progress.bar import IncrementalBar
#from matplotlib import pyplot as plt
from rmsd import kabsch


def get_numpy_V(*args):
    all_v = []
    for emb in args:
        v = emb.V.cpu().numpy()
        if len(args) == 1:
            return v
        all_v.append(v)
    return all_v


# rmsd fast version (imported one is ridiculously slow)
def rmsd_error(rotated_P, Q):
    return np.sqrt(np.sum((rotated_P - Q)**2) / len(Q))


# reimplementation into pytorch
def kabsch_rmsd_error(P, Q):
    rotation = kabsch(P, Q)
    rotated_pmat = P @ rotation
    error = rmsd_error(rotated_pmat, Q)
    return error


# make sure all of the tokens are ordered in same way
def equalize_token_order(all_embs_dict):
    all_tokens_in_common = set()
    for key, hilbert_emb in all_embs_dict.items():
        h_tokens = set(hilbert_emb.dictionary.tokens)
        if len(all_tokens_in_common) == 0:
            all_tokens_in_common = h_tokens
        else:
            all_tokens_in_common.intersection_update(h_tokens)
    print('Tokens in common: {}'.format(len(all_tokens_in_common)))

    # now make sure all rows are in same order
    all_tokens = list(sorted(all_tokens_in_common))
    for key, hilbert_emb in all_embs_dict.items():
        ids = np.array([hilbert_emb.dictionary.get_id(t) for t in all_tokens])
        hilbert_emb.V = hilbert_emb.V[ids] # resort the rows


# compare the big boys
def run_embedding_comparisons(all_embs_dict):
    names = list(sorted(all_embs_dict.keys()))
    n = len(names) # I hate repeating len all the time

    # we will be making a triu matrix of all the rotation errors
    intrinsic_res = np.zeros((n, n))
    rand_res = np.zeros((n, n))
    n_intrinsic_res = np.zeros((n, n)) # normalized
    n_rand_res = np.zeros((n, n)) # normalized

    # iterate over the boys
    bar = IncrementalBar('Doing Kabsch algorithm x4...', max=n**2)
    for i, e1_name in enumerate(names):
        e1_V = get_numpy_V(all_embs_dict[e1_name])
        e1_norms = np.linalg.norm(e1_V, axis=1).reshape(-1, 1)

        for j in range(n):
            e2_V = get_numpy_V(all_embs_dict[names[j]]) # allow compare with self for i==j

            # we will also be comparing to a normally distributed random matrix
            # with the same mean and scaling as those of the other embeddings.
            rand_V = np.random.normal(loc=e2_V.mean(),
                                      scale=e2_V.std(),
                                      size=e2_V.shape)
            rand_norms = np.linalg.norm(rand_V, axis=1).reshape(-1, 1)

            # sanity check
            assert e1_V.shape == e2_V.shape == rand_V.shape, \
                '{} shape: {}; {} shape: {}; rand shape: {}'.format(
                    e1_name, e1_V.shape, names[j], e2_V.shape, rand_V.shape
                )

            # we will be doing comparison with the randomly distributed vecs
            # in each scenario in order to have robust results
            rand_res[i, j] = kabsch_rmsd_error(e1_V, rand_V)
            n_rand_res[i, j] = kabsch_rmsd_error(e1_V / e1_norms, rand_V / rand_norms)

            # compare with the other vecs if j > i (otherwise we already did
            # that computation earlier for when i > j previously in loop)
            if j > i:
                e2_norms = np.linalg.norm(e2_V, axis=1).reshape(-1, 1)
                intrinsic_res[i, j] = kabsch_rmsd_error(e1_V, e2_V)
                n_intrinsic_res[i, j] = kabsch_rmsd_error(e1_V / e1_norms, e2_V / e2_norms)
            bar.next()
    bar.finish()

    # TODO: formalize results serialization.
    np.set_printoptions(precision=4, suppress=True, linewidth=90)
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


# plot rotational error over iterations
def iterative_embedding_comparison(all_embs_dict, relative_to=None):
    errors = []
    prev = relative_to

    bar = IncrementalBar('Doing Kabsch algorithm...', max=len(all_embs_dict))
    for key in sorted(all_embs_dict.keys()):
        bar.next()
        if prev is None:
            prev = all_embs_dict[key]
            continue

        error = kabsch_rmsd_error(*get_numpy_V(prev, all_embs_dict[key]))
        errors.append(error)

        if relative_to is None:
            prev = all_embs_dict[key]
    bar.finish()

    #TODO: formalize results serialization
    plt.figure()
    plt.plot(errors, 'r.-', label='rotational error over time')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('error{}'.format(' (relative to other)' if relative_to is not None else ''))
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare embeddings made by different algorithms.')
    parser.add_argument('emb_path', type=str,
        help='basic string pattern (not regex) for set of embeddings to compare')
    parser.add_argument('-i', '--iterative', action='store_true',
        help='do the comparison over an iteration, rather than comparing to random')
    parser.add_argument('-r', '--relative', type=str,
        help='path to file we would like to compare main the embeddings to')
    m_args = parser.parse_args()

    print('Loading all embeddings starting with pattern: {}...'.format(m_args.emb_path))
    all_embs = {}
    splitted = m_args.emb_path.split('/')
    directory = '/'.join(splitted[:-1])
    pattern = splitted[-1]
    for emb_dname in os.listdir(directory):
        if emb_dname.startswith(pattern):
            all_embs[emb_dname] = load_embeddings('{}/{}'.format(directory, emb_dname), 'cpu')

    # for the specific arguments
    if m_args.iterative:
        relative = load_embeddings(m_args.relative, 'cpu') if m_args.relative else None
        if relative is not None:
            all_embs['rel'] = relative
            equalize_token_order(all_embs)
            all_embs = {k: v for k,v in all_embs.items() if k!='rel'}
        iterative_embedding_comparison(all_embs, relative)
    else:
        equalize_token_order(all_embs)
        run_embedding_comparisons(all_embs)

