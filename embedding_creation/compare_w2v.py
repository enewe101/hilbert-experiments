import shared
import numpy as np
import os
import hilbert as h
import torch
import evaluation.compare_embeddings as c



def compare_embeddings(
    source_dir_name,
    target_dir_name,
    target_embeddings_name='epoch999',
    normalize=False
):
    """
    Using the Kabsch algorithm, find the root mean squared deviation between
    the source embeddings and the target embeddings.  The source embeddings
    for 1000, that is, the source embeddings after a given epoch are compared
    to the target embeddings specified by target_embeddings_name.
    """
    source_dir = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, source_dir_name)
    target_dir = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, target_dir_name)

    target_embeddings_path = os.path.join(target_dir, target_embeddings_name)
    print(target_embeddings_path)
    target_embeddings = h.embeddings.Embeddings.load(target_embeddings_path)
    if normalize:
        target_embeddings.normalize()

    comparison_log = open(os.path.join(target_dir, 'err.txt'), 'w')
    for epoch in range(1000):
        source_embeddings = h.embeddings.Embeddings.load(os.path.join(
            source_dir, 'epoch{}'.format(epoch)))

        if normalize:
            source_embeddings.normalize()
        err = c.kabsch_rmsd_error(
            np.array(source_embeddings.V), np.array(target_embeddings.V))
        print('{}:\t{}'.format(epoch, err))
        comparison_log.write('{}\n'.format(err))
    comparison_log.close()


def compare_final_embeddings(
    source_dir_name,
    target_dir_name,
    source_embeddings_name='epoch999',
    target_embeddings_name='epoch999',
    normalize=False
):
    """
    Using the Kabsch algorithm, find the root mean squared deviation between
    the source embeddings and the target embeddings.  By default, expect the
    source and target embeddings to be contained in a file called epoch999.
    """
    source_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, source_dir_name, 
        source_embeddings_name
    )
    target_path = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, target_dir_name,
        target_embeddings_name
    )

    source_embeddings = h.embeddings.Embeddings.load(source_path)
    target_embeddings = h.embeddings.Embeddings.load(target_path)
    if normalize:
        source_embeddings.normalize()
        target_embeddings.normalize()

    return c.kabsch_rmsd_error(
        np.array(source_embeddings.V),
        np.array(target_embeddings.V)
    )




def check_for_zero_statistics(sample_path, dictionary_path, num_epochs=None):
    """
    See if there are token pairs such that neither a positive nor negative
    sample is every drawn for the token pair.  Provide as output, the fraction
    of such pairs, out of all possible pairs in the vocab.
    """

    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    sample_reader = h.f_delta.SampleReader(sample_path, dictionary)
    stats = torch.zeros((len(dictionary), len(dictionary)))

    num_epochs_started = 0

    while True:
        try:
            for sample in sample_reader:
                for token_id1, token_id2, val in sample:
                    stats[token_id1, token_id2] = 1

        except h.f_delta.NewEpoch:
            num_epochs_started += 1
            print(torch.mean(stats))
            if num_epochs is not None and num_epochs_started > num_epochs:
                break

        except h.f_delta.NoMoreSamples:
            break

    return stats



def run_hilbert_w2v(
    bigram_path,
    out_dir,
    init_embeddings_path,
    learning_rate=0.000001,
    device='cuda',
    times=2000,
    num_epochs=1000,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    bigram = h.bigram.Bigram.load(bigram_path)
    embedder = h.embedder.get_w2v_embedder(
        bigram, 
        shard_factor=1,
        learning_rate=learning_rate,
        verbose=False,
        device=device,
    )

    # Initialize it with vectors used to initialize a real w2v run
    init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)
    embedder.V = init_embeddings.V.clone().to(device)
    embedder.W = init_embeddings.W.clone().to(device)

    # Write initial embeddings to disc before even doing any updates
    out_path = os.path.join(out_dir, 'epoch-init')
    embeddings = h.embeddings.Embeddings(
        embedder.V, embedder.W, bigram.dictionary)
    embeddings.save(out_path)

    # Do 1000 epochs, writing embeddings to disc each time.
    with open(os.path.join(out_dir, 'trace.txt'), 'w') as trace_file:
        for epoch in range(num_epochs):
            embedder.cycle(times)
            out_path = os.path.join(out_dir, 'epoch{}'.format(epoch))
            embeddings = h.embeddings.Embeddings(
                embedder.V, embedder.W, bigram.dictionary)
            embeddings.save(out_path)
            trace_file.write('{}\n'.format(embedder.badness.item()))



def run_hilbert_full_corpus_sample_w2v(
    sample_path,
    out_dir,
    init_embeddings_path,
    learning_rate=0.000001
):
    """
    Runs hbt-w2v, but creates the M and f_delta based on positive and negative
    samples gathered from running std-w2v.  Hence, this hbt-w2v approach 
    tries to simulate more closely std-w2v by using the exact same sample
    statistics.  Similar to ``run_hilbert_sample_based_w2v``, but the sample
    statistics are fully accumulated in order to produce an M-matrix, rather
    than producing updates for each set of k negative and 1 positive samples
    created for a word occurrance.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dictionary = h.dictionary.Dictionary.load(os.path.join(
        sample_path, 'dictionary'))
    Nxx = np.load(os.path.join(sample_path, 'Nxx.npy'))
    Nxx_neg = np.load(os.path.join(sample_path, 'Nxx_neg.npy'))

    delta = h.f_delta.DeltaW2VSamplesFullCorpus(Nxx, Nxx_neg)
    embedder = h.embedder.HilbertEmbedder(
        delta,
        num_vecs=Nxx.shape[0],
        num_covecs=Nxx.shape[0],
        learning_rate=learning_rate,
        shard_factor=1,
        verbose=True
    )

    # Initialize it with vectors used to initialize a real w2v run
    init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)
    embedder.V = init_embeddings.V.clone()
    embedder.W = init_embeddings.W.clone()

    # Arbitrarily designate one epoch to be 200 cycles.
    # Write embeddings to file every epoch
    times = 200

    # Write initial embeddings to disc before even doing any updates
    out_path = os.path.join(out_dir, 'epoch-init')
    embeddings = h.embeddings.Embeddings(embedder.V, embedder.W, dictionary)
    embeddings.save(out_path)

    # Do 1000 epochs, writing embeddings to disc each time.
    for epoch in range(1000):
        embedder.cycle(200)
        out_path = os.path.join(out_dir, 'epoch{}'.format(epoch))
        embeddings = h.embeddings.Embeddings(embedder.V, embedder.W, dictionary)
        embeddings.save(out_path)




def run_hilbert_sample_based_w2v(
    sample_path,
    dictionary_path,
    out_dir,
    init_embeddings_path=None,
    learning_rate=0.025
):
    """
    Runs hbt-w2v, but f_delta is based on positive and negative
    samples gathered from running std-w2v.  Hence, this hbt-w2v approach 
    tries to simulate more closely std-w2v by using the exact same sample
    statistics.  Similar to ``run_hilbert_sample_based_w2v``, but in this case
    individual sets of k negative and 1 positive samples, created for a word
    occurring in the corpus, each produce an update, and these samples are
    "replayed" from a trace of the samples drawn by std-w2v.
    """

    # Create Hilbert Embedder that replays w2v samples
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    sample_reader = h.f_delta.SampleReader(sample_path, dictionary)
    delta = h.f_delta.DeltaW2VSamples(sample_reader)
    embedder = h.embedder.HilbertEmbedder(
        delta,
        num_vecs=len(dictionary),
        num_covecs=len(dictionary),
        learning_rate=learning_rate,
        shard_factor=1,
        verbose=False
    )

    # Initialize it with vectors used to initialize a real w2v run
    if init_embeddings_path is not None:
        init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)
        embedder.V = init_embeddings.V.clone()
        embedder.W = init_embeddings.W.clone()

    # Carry out updates by replaying the samples recorded at sample path
    # After each epoch, save the embeddings to disc.
    epoch_num = 0
    while True:

        try:
            embedder.cycle(None)
        except h.f_delta.NewEpoch:
            pass
        except h.f_delta.NoMoreSamples:
            break

        out_path = os.path.join(out_dir, 'epoch{}'.format(epoch_num))
        embeddings = h.embeddings.Embeddings(embedder.V, embedder.W, dictionary)
        embeddings.save(out_path)

        epoch_num += 1





def run_w2v_replica(
    sample_path,
    dictionary_path,
    out_dir,
    init_embeddings_path=None,
    delay_update=True,
    learning_rate = 0.025,
    device='cpu'
):
    """
    Runs a python implementation of std-w2v.
    """

    # Create a word2vec replica embedder
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    sample_reader = h.f_delta.SampleReader(sample_path, dictionary)
    w2v_replica = h.embedder.W2VReplica(
        sample_reader, delay_update=delay_update, 
        learning_rate=learning_rate, device=device
    )

    # Initialize it with vectors used to initialize a real w2v run
    if init_embeddings_path is not None:
        init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)
        w2v_replica.V = init_embeddings.V.clone().to(device)
        w2v_replica.W = init_embeddings.W.clone().to(device)

    # Carry out updates by replaying the samples recorded at sample path
    # After each epoch, save the embeddings to disc.
    epoch_num = 0
    while True:

        try:
            w2v_replica.cycle(None)
        except h.f_delta.NewEpoch:
            pass
        except h.f_delta.NoMoreSamples:
            break

        out_path = os.path.join(out_dir, 'epoch{}'.format(epoch_num))
        embeddings = h.embeddings.Embeddings(
            w2v_replica.V, w2v_replica.W, dictionary)
        embeddings.save(out_path)

        epoch_num += 1





