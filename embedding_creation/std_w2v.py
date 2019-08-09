import os
import sys
import shared
import argparse
import subprocess
import embedding_creation as ec

def make_w2v_embeddings(
    corpus_path,
    out_dir,
    d=300,
    window=5,
    k=15,
    t_undersample=1,
    alpha_smoothing=0.75,
    processes=1,
    min_count=1,
    seed=None,
    learning_rate=0.025,
    iterations=1000
):

    # Prepare the path to vectors, and make sure output dir exists.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vectors_path = os.path.join(out_dir, 'vectors')

    # Set options that are considered as constants for our purposes.
    mode = 0
    hierarchical_sample = 0
    binary = 0
    cbow = 0 # 0 => skipgram; 1 => cbow

    # Run the word2vec program, using the established options settings.
    command = [
        #'time',
        ec.CONSTANTS.W2V_EXECUTABLE_PATH,
        '-train', corpus_path,
        '-output', vectors_path,
        '-cbow', str(cbow),
        '-size', str(d),
        '-window', str(window),
        '-negative', str(k),
        '-hs', str(hierarchical_sample),
        '-sample', str(t_undersample),
        '-power', str(alpha_smoothing),
        '-threads', str(processes),
        '-binary', str(binary), 
        '-min-count', str(min_count),
        '-alpha', str(learning_rate),
        '-iter', str(iterations)
    ]

    # Apply the seed if given
    if seed is not None:
        command.extend(['-seed', str(seed)])

    print(' '.join(command))
    return subprocess.run(command)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description=("Runs standard w2v"))

    parser.add_argument(
        '--corpus_path', '-c', required=True, 
        help="Filename for corpus to be used"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, 
        help="Directory in which to write embeddings."
    )
    parser.add_argument(
        '--processes', '-p', type=int, default=1, 
        help="Number of processes to use for parallelization."
    )
    parser.add_argument(
        '--seed', '-s', type=int, required=True, help="Random seed."
    )
    parser.add_argument(
        '--learning-rate', '-l', type=float, required=True,
        help="Learning rate."
    )
    parser.add_argument(
        '--iterations', '-i', type=int, default=5, 
        help="Number of iterations (epochs) to run."
    )

    # Hyperparams
    parser.add_argument(
        '--t-undersample', '-t', type=float, required=True,
        help="Post-sampling (clean) Common word undersampling threshold"
    )
    parser.add_argument(
        '--alpha-smoothing', '-a', type=float, default=0.75,
        help="Exponent used for unigram smoothing."
    )
    parser.add_argument(
        '--k', '-k', type=int, default=15,
        help="Exponent used for unigram smoothing."
    )

    args = vars(parser.parse_args())
    args['corpus_path'] = corpus_path = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['corpus_path'])
    args['out_dir'] = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, args['out_dir'])

    make_w2v_embeddings(**args)

