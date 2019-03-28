import os
import time
import hilbert as h
import shared
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
try:
    import torch
except ImportError:
    torch = None




#
#   Plotting
#

def do_plot(glove_results_dir, pmi_path, out_path):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5.3))
    large_fontsize = 24
    fontsize = 24
    tick_fontsize = 22
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

    plot_visible_pmi(
        pmi_path, ax=ax1,
        line_args={'color':'0.5'},
        mean_args={'color':'0.0'},
        text_args={'fontsize':tick_fontsize, 'ha':'center'}
    )
    ax1.set_xlabel('$\mathrm{PMI}(i,j)$', fontsize=fontsize)
    ax1.set_ylabel('Number of pairs $(i,j)$', fontsize=fontsize, labelpad=12)

    y = [250000, 200000, 150000, 100000, 50000]
    labels = ['250k', '200k', '150k', '100k', '50k']
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=tick_fontsize)
    plt.setp(ax1.get_yticklabels(), fontsize=tick_fontsize)
    plt.setp(ax1.get_xticklabels(), fontsize=tick_fontsize)

    analyze_glove_bias(
        glove_results_dir, ax2,
        scatter_args={'color':'0.50'},
        line_args={'color':'0.0'},
        text_args={'rotation':45, 'fontsize':tick_fontsize}
    )
    plt.setp(ax2.get_yticklabels(), fontsize=tick_fontsize)
    plt.setp(ax2.get_xticklabels(), fontsize=tick_fontsize)

    xlabel = r'$\lg\frac{N_i}{\sqrt{N}}$'
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ylabel = '$b_i$'
    ax2.set_ylabel(ylabel, fontsize=fontsize, labelpad=-1)

    ax1.text(-12.5, 270000, 'A)', fontsize=large_fontsize)
    ax2.text(-1.1, 8.5, 'B)', fontsize=large_fontsize)

    plt.subplots_adjust(
        left=0.14, bottom=0.2, right=0.96, top=0.96, wspace=0.30, hspace=None
    )
    plt.savefig(out_path)

    plt.show()




#
#   Glove Bias Plotting
#

def analyze_glove_bias(
    glove_results_dir, ax=None, scatter_args=None, line_args=None,
    text_args=None
):
    line_args = line_args or {}
    scatter_args = scatter_args or {}
    text_args = text_args or {}
    in_path = glove_results_dir

    fields = read_bias_frequency_pairs(in_path)
    v_biases, w_biases, log_freqs_normed, log_freqs = fields

    if ax is None:
        plottable = plt
    else:
        plottable = ax

    plottable.scatter(log_freqs_normed, v_biases, s=10, **scatter_args)
    plottable.plot([-1,9], [-1,9], **line_args)
    plottable.text(5.0, 6, r'$b_i = \lg\frac{N_i}{\sqrt{N}}$', **text_args)

    if ax is None:
        plt.show()

    return plottable



    #mean_biases = [
    #    (v_bias + w_bias) / 2 for v_bias, w_bias in zip(v_biases, w_biases)]
    #plt.scatter(log_freqs, v_biases, s=10)
    #plt.plot(log_freqs, mean_biases)



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




#
#   PMI Plotting
#
def plot_visible_pmi(
        path, ax=None, line_args=None, mean_args=None, text_args=None
    ):
    line_args = line_args or {}
    mean_args = mean_args or {}
    bin_centers, n = read_pmi_histogram(path)
    total = sum([
        bin_center * n_item for bin_center, n_item in zip(bin_centers, n)])
    mean = total / sum([n_item for n_item in n])
    if ax is None:
        plottable = plt
    else:
        plottable = ax
    plottable.plot(bin_centers, n, **line_args)
    plottable.set_ylim(-10000,300000)
    plottable.plot([mean, mean], [0,250000], **mean_args)
    plottable.text(
        mean, 260000, str('mean = %.2f' % mean), **text_args)
    if ax is None:
        plottable.show()
    return plottable


def read_pmi_histogram(path):
    bin_centers = []
    n = []
    with open(path) as in_file:
        for line in in_file:
            line = line.strip()
            if line == '':
                continue
            bin_center_str, n_str = line.split()
            bin_centers.append(float(bin_center_str))
            n.append(int(n_str))
    return bin_centers, n



def calc_visible_pmi_histogram(bigram_or_path, out_path=None, device=None):
    if isinstance(bigram_or_path, str):
        print('loading bigram...')
        bigram_path = bigram_or_path
        bigram = h.bigram.BigramBase.load(bigram_path, device=device)
    elif isinstance(bigram_or_path, h.bigram.BigramBase):
        bigram = bigram_or_path
    else: raise ValueError(
        "First to argument to calc_visible_pmi_histogram must be a bigram "
        "or path to bigram data"
    )

    print('calculating visible pmi values...')
    visible_pmi = calc_visible_pmi(bigram).reshape(-1)
    n, bins = np.histogram(visible_pmi, bins='auto')
    bin_centers = [ 0.5*(bins[i]+bins[i+1]) for i in range(len(n))]
    if out_path is None:
        return bin_centers, n

    print('writing results to disk...')
    with open(out_path, 'w') as out_file:
        out_file.write(''.join([
            '{}\t{}\n'.format(bin_center, n_item) 
            for bin_center, n_item in zip(bin_centers, n)
        ]))



if __name__ == '__main__':
    an.do_plot(
        '/Users/enewel3/projects/hilbert/data/glove-analysis/'
        'std-glv-v10k-iter10-x100-bias',
        '/Users/enewel3/projects/hilbert/data/pmi_plottable/pmi-hist-40k.tsv',
        '/Users/enewel3/projects/hilbert/data/glove-analysis/glove-bias-pmi.pdf'
    )
