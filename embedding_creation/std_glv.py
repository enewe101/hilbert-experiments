import embedding_creation as ec
import os
import sys
import shared
import argparse
import subprocess



MIN_COUNT = 17921 # For gigaword this provides a vocab of 10001


def make_glove_embeddings(
    out_dir,
    min_count=5,
    dimension=300,
    iterations=15,
    window=5,
    threads=8,
    X_max=100,
    save_mode=0,
    stats_path=None,
):
    corpus = shared.CONSTANTS.TOKENIZED_CAT_FULL_PATH
    command = [
        ec.CONSTANTS.LOCAL_GLOVE_EXECUTABLE_PATH,
        corpus,
        out_dir, 
        ec.CONSTANTS.GLOVE_EXECUTABLE_PATH,
        str(min_count),
        str(dimension),
        str(iterations),
        str(window),
        str(threads),
        str(X_max),
        str(save_mode),
        str(stats_path)
    ]

    print(str(save_mode))

    trace = ' '.join(command)
    print(trace)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'trace.txt'), 'w') as trace_file:
        trace_file.write(trace)
    return subprocess.run(command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run GloVe")
    parser.add_argument(
        '--out-dir', '-o', type=str, required=True,
        help='Directory at which to save embeddings'
    )
    parser.add_argument(
        '--min-count', '-m', type=int, default=0,
        help='Drop tokens that occur fewer than min-count times'
    )
    parser.add_argument(
        '--iterations', '-i', type=int, default=15, help='Number of iterations'
    )
    parser.add_argument(
        '--threads', '-t', type=int, required=True, help="Number of threads"
    )
    parser.add_argument(
        '--X-max', '-x', type=float, default=100., help="X_max hyperparameter"
    )
    parser.add_argument(
        '--dimension', '-d', type=int, default=300., 
        help="Vector dimensionality"
    )
    parser.add_argument(
        '--save-mode', '-s', type=int, default=2, 
        help=(
            "Whether to output (0) vectors, covectors, and biases, (1) only "
            "vectors (no biases or covectors), or (2) vectors and covectors "
            "(no biases)."
        )
    )

    parser.add_argument(
        '--window', '-w', type=int, default=None,
        help='Cooccurrence window size'
    )
    parser.add_argument(
        '--stats-path', '-p', type=str, default='', 
        help=(
            "Path to the directory containing the unigram and bigram data "
            "files previously extracted by glove.  If not provided, GloVe "
            "will extract those files from the corpus.  If provided, GloVe "
            "will skip the extraction step and use the already-extracted data."
        )
    )

    args = vars(parser.parse_args())

    # Either stats path must be set, or window size must be set
    if args['stats_path'] == '' and args[window] is None:
        raise ValueError(
            "Either stats path must be set, or window size must be set")

    # outdir is relative to the embeddings directory
    args['out_dir'] = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, args['out_dir'])

    # stats_path is relative to embeddings directory.
    if args['stats_path'] is not '':
        args['stats_path'] = os.path.join(
            shared.CONSTANTS.EMBEDDINGS_DIR, args['stats_path'])

    make_glove_embeddings(**args)

