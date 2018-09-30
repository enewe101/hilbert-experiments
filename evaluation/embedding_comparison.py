import argparse
import rmsd
import numpy as np
import os
from evaluation.run_experiments import load_embeddings
from progress.bar import IncrementalBar


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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare embeddings made by different algorithms.')
    parser.add_argument('emb_path', type=str,
        help='basic string pattern (not regex) for set of embeddings to compare')
    args = parser.parse_args()

    print('Loading all embeddings starting with pattern: {}...'.format(args.emb_path))
    all_embs = {}
    splitted = args.emb_path.split('/')
    directory = '/'.join(splitted[:-1])
    pattern = splitted[-1]
    for emb_dname in os.listdir(directory):
        if emb_dname.startswith(pattern):
            all_embs[emb_dname] = load_embeddings('{}/{}'.format(directory, emb_dname))
    run_embedding_comparisons(all_embs)

