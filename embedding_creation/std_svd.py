import os
import time
import shared
import argparse
import numpy as np
import hilbert as h

def make_svd_embeddings(
    bigram_path, 
    out_path,
    k=15,
    alpha=3/4.,
    t_clean=1.,
    clip_after_shift=False
):

    start = time.time()

    print("Starting SVD on {}.".format(bigram_path))
    print("Writing to {}. k={}. alpha={}. t={}. clip_after={}. ".format(
        out_path, k, alpha, t_clean, clip_after_shift))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    bigram = h.bigram.Bigram.load(bigram_path, device='cpu')

    U, s, V = h.embedder.calc_svm_embeddings(
        bigram, k=k, alpha=alpha, t_clean=t_clean, 
        clip_after_shift=clip_after_shift
    )

    np.save(os.path.join(out_path, 'U.npy'), U)
    np.save(os.path.join(out_path, 's.npy'), s)
    np.save(os.path.join(out_path, 'V.npy'), V)

    print("Total wall time: {}".format(time.time()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        "Create embeddings using SVD on SPPMI matrix based on supplied bigram "
        "Statistics."
    ))
    parser.add_argument(
        '--bigram-dirname', '-b', required=True, dest='bigram_path',
        help="Path to directory containing bigram statistics."
    )
    parser.add_argument(
        '--out-dirname', '-o', required=True, dest='out_path',
        help="Path at which to save embeddings."
    )
    parser.add_argument(
        '-k', default=15, type=int, 
        help="Shift elements in ppmi matrix by -ln(k)."
    )
    parser.add_argument(
        '--clip-after-shift', '-c', action='store_true',
        help="Clip negative values after shifting (default clip before)."
    )
    parser.add_argument(
        '--alpha', '-a', type=float, default=3/4.,
        help="Exponent used for unigram distribution smoothing."
    )
    parser.add_argument(
        '--t-clean', '-t', type=float, default=None,
        help="Exponent used for unigram distribution smoothing."
    )

    args = vars(parser.parse_args())
    args['bigram_path'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['bigram_path'])
    args['out_path'] = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, args['out_path'])

    make_svd_embeddings(**args)

