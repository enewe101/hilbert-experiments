import hilbert as h
import numpy as np
import pdb
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.switch_backend('agg')


W2V_K = 1
W2V_T = 2e-5

### Helper functions! ###
def build_debug_samples(path):
    print("Loading bigrams...")
    bigrams = h.bigram.Bigram.load(path, device='cpu')

    print("Gloving bigrams...")
    gf_delta = h.f_delta.DeltaGlove(bigrams, device='cpu')
    gf_delta.load_shard(None, None)
    glove_mults = gf_delta.multiplier.flatten().numpy()

    print('W2Vecing bigrams...')
    w2v_delta = h.f_delta.DeltaW2V(bigrams, W2V_K, device='cpu')
    w2v_delta.load_shard(None, None)
    w2v_mults = w2v_delta.multiplier.flatten().numpy()

    print('Building original Nxx...')
    nxx = np.array(bigrams.Nxx.todense()).flatten()

    print('W2Vecing undersampled bigrams...')
    bigrams.apply_w2v_undersampling(W2V_T)
    under_w2v_delta = h.f_delta.DeltaW2V(bigrams, W2V_K, device='cpu')
    under_w2v_delta.load_shard(None, None)
    under_w2v_mults = under_w2v_delta.multiplier.flatten().numpy()

    print('Building undersampled Nxx...')
    under_nxx = np.array(bigrams.Nxx.todense()).flatten()

    print('Filtering and serializing data...')
    items = [('glove_mults', glove_mults),
             ('w2v_mults-k1', w2v_mults),
             ('under_w2v_mults-k1', under_w2v_mults),
             ('nxx', nxx),
             ('under_nxx-k1', under_nxx)]

    # set the random seed and filter the data to 1% (this is 1% of 10,000,000, so things are still big!
    np.random.seed(1917)
    rand_idx = np.random.choice(np.arange(len(nxx)), size=int(len(nxx) * 0.01))
    for name, arr in items:
        np.save(f'debug/{name}', arr[rand_idx])


def load_debug_samples():
    d = {}
    print('Loading debug samples...')
    for f in os.listdir('debug'):
        if f.endswith('.npy'):
            d[f.rstrip('.npy')] = np.load(f'debug/{f}')
    print('Done loading!')
    return d


def get_png_idx(prefix):
    png_idx = 0
    for f in os.listdir('png'):
        if f.startswith(prefix):
            idx = int(f.split('_')[1].rstrip('.png'))
            png_idx = max(idx, png_idx)
    return png_idx + 1


def save_figure(fig, prefix):
    assert '_' not in prefix
    png_idx = get_png_idx(prefix)
    fig.savefig(f'png/{prefix}_{png_idx}.png', bbox_inches='tight')
    print(f'Plotted png/{prefix}_{png_idx}...')
    plt.clf()
### /end ###

def examine_undersample(data_dict):
    fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(16, 10))

    under_w2v_mults = np.log(data_dict['under_w2v_mults'] / max(data_dict['under_w2v_mults']))
    nxx = np.log(data_dict['nxx'])
    under_nxx = np.log(data_dict['under_nxx'])
    idx = nxx != np.inf

    axs[0].plot(nxx[idx], under_w2v_mults[idx], 'r.', alpha=0.75)
    axs[0].set_xlabel('Log Nxx')
    axs[0].set_ylabel('Log Undersampled W2V multiplier')
    axs[0].ticklabel_format(style='plain')
    axs[0].set_ylim(-14., 0.1)
    axs[0].set_xlim(-4., 17.5)

    axs[1].plot(under_nxx[idx], under_w2v_mults[idx], 'm.', alpha=0.75)
    axs[1].set_xlabel('Log Nxx (Undersampled Nxx)')
    axs[1].set_ylabel('Log Undersampled W2V multiplier')
    axs[1].ticklabel_format(style='plain')
    axs[1].set_ylim(-14., 0.1)
    axs[1].set_xlim(-4., 17.5)

    png_idx = get_png_idx('examine-under')
    fig.savefig(f'png/examine-under_{png_idx}', bbox_inches='tight')
    print(f'Plotted png/examine-under_{png_idx}...')
    plt.clf()


def examine_glove(data_dict):
    nxx = data_dict['nxx']
    glove_mults = data_dict['glove_mults']
    w2v_mults = data_dict['w2v_mults']
    uw2v_mults = data_dict['under_w2v_mults']

    fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(16, 8))
    glove_mults /= max(glove_mults) # this will be 2
    w2v_mults /= max(w2v_mults)
    uw2v_mults /= max(uw2v_mults)

    axs[0].plot(nxx, glove_mults, 'b.', alpha=0.75, label='Glove')
    axs[0].plot(nxx, w2v_mults, 'g.', alpha=0.75, label='W2V')
    axs[0].plot(nxx, uw2v_mults, 'r.', alpha=0.75, label='Under W2v')
    axs[0].plot([100, 100], [0, 1], 'k--', alpha=0.5) # plot vertical bar
    axs[0].set_xlabel('Nxx')
    axs[0].set_ylabel('Multiplier, f(Nxx)')
    axs[0].legend()
    axs[0].ticklabel_format(style='plain')

    axs[1].plot(nxx, glove_mults, 'b.', alpha=0.75, label='Glove')
    axs[1].plot(nxx, w2v_mults, 'g.', alpha=0.75, label='W2V')
    axs[1].plot(nxx, uw2v_mults, 'r.', alpha=0.75, label='Under W2v')
    axs[1].plot([100, 100], [0, 1], 'k--', alpha=0.5) # plot vertical bar
    axs[1].set_xlabel('Nxx')
    axs[1].set_ylabel('Glove multiplier, f(Nxx)')
    axs[1].set_xlim(-5., 200.)
    save_figure(fig, 'upclose')


def overlay_log_multipliers(data_dict):
    fig, axs = plt.subplots(1, 1, tight_layout=False, figsize=(20, 12))
    nxx = data_dict['nxx']
    idx = nxx != 0
    logf = lambda x: np.log(x / max(x))
    nxx = logf(nxx)[idx]
    gm = logf(data_dict['glove_mults'])[idx]
    w2v = logf(data_dict['w2v_mults'])[idx]
    uw2v = logf(data_dict['under_w2v_mults'])[idx]
    plots = [(nxx, gm, 'Log GloVe', 'b'),
             (nxx, w2v, 'Log W2V', 'g'),
             (nxx, uw2v, 'Log Under W2V', 'r')]

    for nxx, vals, name, color in plots:
        axs.plot(nxx, vals, f'{color}.', alpha=0.5, label=name)
    axs.legend()
    axs.set_xlabel('Log normalized Nxx')
    axs.set_ylabel('Log normalized multiplier')
    save_figure(fig, 'overlay-log-multipliers')


def plot_multipliers(data_dict, log_nxx=True, log_glove_mults=True, log_w2v_mults=True):
    fig, axs = plt.subplots(3, 1, tight_layout=False, figsize=(16, 10))

    # set up some basic functions
    glv_log_fun = lambda x: np.log(x / max(x)) if log_glove_mults else x
    w2v_log_fun = lambda x: np.log(x / max(x)) if log_w2v_mults else x
    nxx_log_fun = lambda x: np.log(x) if log_nxx else x
    glv_log_str = lambda s: f'log {s}' if log_glove_mults else s
    w2v_log_str = lambda s: f'log {s}' if log_w2v_mults else s
    nxx_log_str = lambda s: f'log {s}' if log_nxx else s

    plots = [(nxx_log_fun(data_dict['nxx']), glv_log_fun(data_dict['glove_mults']), glv_log_str('glove'), 'b'),
             (nxx_log_fun(data_dict['nxx']), w2v_log_fun(data_dict['w2v_mults']), w2v_log_str('w2v'), 'g'),
             (nxx_log_fun(data_dict['nxx']), w2v_log_fun(data_dict['under_w2v_mults']), w2v_log_str('under w2v'), 'r')]

    idx = plots[0][0] != -np.inf
    for axidx, (nxx, vals, name, color) in enumerate(plots):
        axs[axidx].plot(nxx[idx], vals[idx], f'{color}.', alpha=0.75)
        axs[axidx].set_xlabel(nxx_log_str('Nxx'))
        axs[axidx].set_ylabel(f'{name} multiplier')
        axs[axidx].ticklabel_format(style='plain')
        if log_nxx:
            axs[axidx].set_xlim(-4., 17.5)
        if log_w2v_mults and log_glove_mults:
            axs[axidx].set_ylim(-17.5, 0.5)

    save_figure(fig, 'multipliers-to-nxx')


def plot_histogram(glove_mults, w2v_mults, under_w2v_mults, log_glove=False, log_w2v=True):
    # set up some basic constants
    glv_log_fun = lambda x: np.log(x) if log_glove else x
    w2v_log_fun = lambda x: np.log(x) if log_w2v else x
    glv_log_str = lambda s: f'log {s}' if log_glove else s
    w2v_log_str = lambda s: f'log {s}' if log_w2v else s
    glv_idx = glove_mults > 0 if log_glove else np.arange(len(glove_mults))

    # check for what we're zeroing out
    gm = glove_mults[glv_idx]
    if len(gm) < len(glove_mults):
        print('Warning: Glove is removing {:2.2f}% of samples due to -infs'.format(
              100 - 100 * len(gm) / len(glove_mults)))

    print('Plotting histograms...')
    fig, axs = plt.subplots(3, 1, tight_layout=False, figsize=(16, 10))

    # plot them
    strs = [(glv_log_fun(gm / max(gm)), glv_log_str('GloVe Multipliers'), 'b'),
            (w2v_log_fun(w2v_mults / max(w2v_mults)), w2v_log_str('W2V Multipliers'), 'g'),
            (w2v_log_fun(under_w2v_mults / max(under_w2v_mults)), w2v_log_str('Under. W2V Multipliers'), 'r')]

    for axidx, (vals, xlab, color) in enumerate(strs):
        n_, bins_, patches_ = axs[axidx].hist(vals, 50, density=True, facecolor=color, alpha=0.75)
        axs[axidx].set_ylabel('Density')
        axs[axidx].set_xlabel(xlab)
        axs[axidx].grid(True)
        # if axidx > 0 and log_w2v:
        #     axs[axidx].set_xlim(-0.1, 20.)

    # get the PNG idx
    save_figure(fig, 'histograms')


##########
def main():
    # build_debug_samples('/home/rldata/hilbert-embeddings/cooccurrence/5w-dynamic-10k/thresh1')
    data_dict = load_debug_samples()

    # examine_undersample(data_dict)

    # build all the possible histograms
    # for lgglv in [True, False]:
    #     for lgwv in [True, False]:
    #         plot_histogram(data_dict['glove_mults'], data_dict['w2v_mults'], data_dict['under_w2v_mults'],
    #                        log_glove=lgglv, log_w2v=lgwv)

    # build all possible plots of the multipliers
    # for lgglv in [True, False]:
    #     for lgwv in [True, False]:
    #         for lgnxx in [True, False]:
    #             plot_multipliers(data_dict, log_glove_mults=lgglv, log_w2v_mults=lgwv, log_nxx=lgnxx)

    # plot_multipliers(data_dict, log_glove_mults=True, log_w2v_mults=True, log_nxx=True)

    examine_glove(data_dict)

    # overlay_log_multipliers(data_dict)

if __name__ == '__main__':
    main()

