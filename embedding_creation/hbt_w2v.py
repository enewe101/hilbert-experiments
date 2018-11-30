import os
import sys
import time
import argparse

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


def run_w2v(

    # Main options
    bigram_path,
    save_embeddings_dir,
    init_embeddings_path=None,
    seed=0,

    # Training period options 
    iters_per_epoch=100,
    epochs=100,

    # Debug Option
    small=False,

    # Embedder Options
    k=15,
    alpha=3./4,
    t_clean=None,
    d=300,
    update_density=1,
    solver='sgd',
    shard_factor=1,
    learning_rate=1e-5,
    verbose=True,
    device=None,
):
    """
    `init_embeddings_path`  [ str | None ]: load embeddings from the given or,
        if None, use random embeddings.
    """

    if small:
        print('small not yet implemented')
    print(update_density)

    # Set the random seed.
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load bigrams
    start = time.time()
    bigram = h.bigram.Bigram.load(bigram_path)
    print('bigrams loading time {}'.format(time.time() - start))

    # Get initial embeddings (if any)
    init_vecs = None
    if init_embeddings_path is not None:
        init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)
        init_vecs = (init_embeddings.V, init_embeddings.W)

    # Make the embedder
    embedder, solver = h.embedder.get_w2v_embedder(
        bigram=bigram,
        k=k,
        alpha=alpha,
        t_clean=t_clean,
        d=d,
        update_density=update_density,
        solver=solver,
        shard_factor=shard_factor,
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        verbose=verbose,
        device=device
    )

    # Work out the path at which embeddings will be saved.
    if not os.path.exists(save_embeddings_dir):
        os.makedirs(save_embeddings_dir)

    # Write a description of this run within the embeddings save directory
    trace_path = os.path.join(save_embeddings_dir,'trace.txt')
    with open(trace_path, 'w') as trace_file:
        trace_file.write(describe(embedder, solver))

        # Finally run many training periods, saving embeddings every period.
        for epoch in range(epochs):

            # Execute one training epoch.
            print('epoch\t{}'.format(epoch))
            solver.cycle(times=iters_per_epoch)

            # Save a copy of the embeddings.
            embeddings = h.embeddings.Embeddings(
                V=embedder.V, W=embedder.W, dictionary=bigram.dictionary)
            embeddings_save_path = os.path.join(
                save_embeddings_dir,
                'iter-{}'.format(iters_per_epoch * epoch)
            )
            embeddings.save(embeddings_save_path)
            trace_file.write('iter\t{}learning_rate\t{}\tloss\t{}\n'.format(
                iters_per_epoch * epoch, embedder.learning_rate, 
                embedder.badness
            ))

    return embedder, solver


def describe(embedder, solver):
    description = ""
    description += "\n> embedder\n"
    description += "f_delta\t{}\n".format(embedder.delta.__class__.__name__)
    description += "d\t{}\n".format(embedder.d)
    description += "learning_rate\t{}\n".format(embedder.learning_rate)
    description += "one_sided\t{}\n".format(embedder.one_sided)
    constrainer_name = (
        'None' if not embedder.constrainer else embedder.constrainer.__name__)
    description += "constrainer\t{}\n".format(constrainer_name)
    description += "shard_factor\t{}\n".format(embedder.shard_factor)
    description += "device\t{}\n".format(embedder.device)

    f_delta = embedder.delta
    description += "\n> f_delta\n"
    description += "k\t{}\n".format(f_delta.k)

    #M = f_delta.M
    #description += "\n> M\n"
    #description += "base\t{}\n".format(M.__class__.__name__)
    #description += "shift_by\t{}\n".format(M.shift_by)
    #description += "neg_inf_val\t{}\n".format(M.neg_inf_val)
    #description += "clip_thresh\t{}\n".format(M.clip_thresh)
    #description += "diag\t{}\n".format(M.diag)

    description += "\n> solver\n"
    description += "solver\t{}\n".format(solver.__class__.__name__)
    description += "solver\t{}\n".format(solver.learning_rate)
    try:
        description += "solver\t{}\n".format(solver.momentum_decay)
    except AttributeError:
        pass

    return description



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Runs a hilbert embedder that simulates word2vec's objective and loss"))
    parser.add_argument(
        '--bigram', '-b', required=True, dest='bigram_path',
        help="Name of the bigrams subdirectory containing bigram statistics"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, dest='save_embeddings_dir',
        help="Name of embeddings subdirectory in which to store embeddings"
    )
    parser.add_argument(
        '--init', '-i', dest="init_embeddings_path", default=None,
        help="Name of embeddings subdirectory to use as initialization"
    )
    parser.add_argument(
        '--seed', '-S', type=int, required=True, help="Random seed"
    )
    parser.add_argument(
        '--solver', '-s', default='sgd', help="Type of solver to use"
    )
    parser.add_argument(
        '--learning-rate', '-l', type=float, required=True, help="Learning rate"
    )
    parser.add_argument(
        '--t-clean', '-t', type=float, default=None, 
        help="Post-sampling (clean) Common word undersampling threshold"
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, 
        help="Number of epochs to run.  Embeddings are saved after each epoch."
    )
    parser.add_argument(
        '--iters-per-epoch', '-I', type=int, default=100, 
        help="Number of iterations per epoch"
    )
    parser.add_argument(
        '--update-density', '-u', type=float, default=1, 
        help="density of random bitmask applied to Delta before update"
    )
    parser.add_argument(
        '--small', '-m',  action="store_true", 
        help="run with a small mock bigram."
    )


    # Parse the arguments
    args = vars(parser.parse_args())

    # For convenience, paths are relative to dedicated subdirectories in the 
    # hilbert data folder.
    args['save_embeddings_dir'] = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, args['save_embeddings_dir'])
    args['bigram_path'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['bigram_path'])
    if args['init_embeddings_path'] is not None:
        args['init_embeddings_path'] = os.path.join(
            shared.CONSTANTS.EMBEDDINGS_DIR, args['init_embeddings_path'])

    run_w2v(**args)

