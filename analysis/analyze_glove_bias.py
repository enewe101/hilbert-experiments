import os
import time
import hilbert as h
import shared
from matplotlib import pyplot as plt
try:
    import torch
except ImportError:
    torch = None


def analyze_glove_bias(glove_results_dir):
    in_path = os.path.join(
        shared.CONSTANTS.GLOVE_ANALYSIS_DIR,
        os.path.basename(glove_results_dir)
    )
    fields = read_bias_frequency_pairs(in_path)
    v_biases, w_biases, log_freqs_normed, log_freqs = fields
    #mean_biases = [
    #    (v_bias + w_bias) / 2 for v_bias, w_bias in zip(v_biases, w_biases)]
    plt.scatter(log_freqs_normed, v_biases, s=10)
    plt.scatter(log_freqs, v_biases, s=10)
    plt.plot([-1,9], [-1,9])
    #plt.plot(log_freqs, mean_biases)
    plt.show()


def prepare_bias_frequency_pairs(bigram_path, glove_results_dir, force=False):

    out_path = os.path.join(
        shared.CONSTANTS.GLOVE_ANALYSIS_DIR,
        os.path.basename(glove_results_dir)
    )

    # Just read bias frequency pairs from disk if they are there.
    if not force and os.path.exists(out_path):
        return read_bias_frequency_pairs(out_path)

    # Otherwise, calculate them.
    v_biases, w_biases, log_freqs_normed, log_freqs = calc_bias_frequency_pairs(
        bigram_path, glove_results_dir)

    # And write them to disk for next time
    write_bias_frequency_pairs(
        v_biases, w_biases, log_freqs_normed, log_freqs, out_path)

    return v_biases, w_biases, log_freqs_normed, log_freqs


def read_bias_frequency_pairs(in_path):
    v_biases, w_biases, log_freqs_normed, log_freqs = [], [], [], []
    with open(in_path) as in_file:
        first = True
        for line in in_file:
            if first:
                first=False
                continue
            fields = line.strip().split('\t')
            v_bias_str, w_bias_str, log_freq_normed_str, log_freq_str = fields
            v_biases.append(float(v_bias_str))
            w_biases.append(float(w_bias_str))
            log_freqs_normed.append(float(log_freq_normed_str))
            log_freqs.append(float(log_freq_str))
    return v_biases, w_biases, log_freqs_normed, log_freqs


def calc_bias_frequency_pairs(bigram_path, glove_results_dir):

    print('loading bigrams...')
    start = time.time()
    bigram = h.bigram.BigramBase.load(bigram_path, device='cpu')

    print('loading embeddings...')
    embeddings_dir = os.path.join(glove_results_dir, 'vectors')
    embeddings = h.embeddings.Embeddings.load(embeddings_dir, device='cpu')

    # Make the bigram statistics and embeddings use a common token ordering
    print('sorting embedding...')
    embeddings.sort_by_tokens(bigram.dictionary, allow_mismatch=True)

    print('calculating log frequencies...')
    v_biases = [
        embeddings.v_bias[embeddings.dictionary.get_id(token)].item()
        for token in embeddings.dictionary.tokens
    ]
    w_biases = [
        embeddings.w_bias[embeddings.dictionary.get_id(token)].item()
        for token in embeddings.dictionary.tokens
    ]
    log_freqs_normed = [
        torch.log(
            bigram.uNx[bigram.dictionary.get_id(token),0] 
            / torch.sqrt(bigram.uN)
        ).item()
        for token in embeddings.dictionary.tokens
    ]
    log_freqs = [
        torch.log(bigram.uNx[bigram.dictionary.get_id(token),0]).item()
        for token in embeddings.dictionary.tokens
    ]

    return v_biases, w_biases, log_freqs_normed, log_freqs


def write_bias_frequency_pairs(
    v_biases, w_biases, log_freqs_normed, log_freqs, out_path
):
    with open(out_path, 'w') as out_file:
        out_file.write('v_bias\tw_bias\tlg(Nx/N)\tlg(Nx)\n')
        out_file.write('\n'.join([
            '{}\t{}\t{}\t{}'.format(v_bias, w_bias, log_freq_normed, log_freq) 
            for v_bias, w_bias, log_freq_normed, log_freq
            in zip(v_biases, w_biases, log_freqs_normed, log_freqs)
        ]))


