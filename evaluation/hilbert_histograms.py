import hilbert as h
import numpy as np
import pdb
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.switch_backend('agg')


def load_debug_samples():
    print('Loading debug...')
    arr = np.load('debug_samples.npy')
    print('Done loading debug...')
    return arr[0], arr[1], arr[2], arr[3]


def save_debug_samples(g_nxx, glove_vals, w_nxx, w2v_vals, prop=0.05):
    assert len(g_nxx) == len(w_nxx)
    idx = np.random.choice(len(g_nxx), size=int(len(g_nxx) * prop))
    arr = np.array([g_nxx[idx], glove_vals[idx],
                    w_nxx[idx], w2v_vals[idx]])
    np.save(f'debug_samples', arr)


def histogram(path, plot_histos=True, plot_mults=True, load_debug=False):

    if load_debug:
        dstr = '_debug1'
        g_nxx, glove_vals, w_nxx, w2v_vals = load_debug_samples()
    else:
        # get the glove results
        bigrams = h.bigram.Bigram.load(path, device='cpu')
        dstr = '_1-1'
        if plot_mults:
            g_nxx = np.array(bigrams.Nxx.todense()).flatten()
        gf_delta = h.f_delta.DeltaGlove(bigrams, device='cpu')
        gf_delta.load_shard(None, None)
        glove_vals = gf_delta.multiplier.flatten().numpy()

        # get the w2v results
        w2v_k = 15
        w2v_t = 2e-5
        bigrams.apply_w2v_undersampling(w2v_t)
        if plot_mults:
            w_nxx = np.array(bigrams.Nxx.todense()).flatten()
        wf_delta = h.f_delta.DeltaW2V(bigrams, w2v_k, device='cpu')
        wf_delta.load_shard(None, None)
        w2v_vals = wf_delta.multiplier.flatten().numpy()
        if plot_mults:
            print('Serializing debug data...')
            save_debug_samples(g_nxx, glove_vals, w_nxx, w2v_vals)

    # plot the multiplier vs nxx for both
    if plot_mults:
        print('Plotting multipliers vs nxx...')
        fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(16, 8))

        # plot them
        strs = [(g_nxx, glove_vals, 'glove', 'b'),
                (w_nxx, w2v_vals, 'w2v', 'g')]

        for axidx, (nxx, vals, name, color) in enumerate(strs):
            axs[axidx].plot(nxx, vals, f'{color}.')
            axs[axidx].set_xlabel('Nxx')
            axs[axidx].set_ylabel(f'{name} multiplier')

        fig.savefig(f'png/NXX_VALUES{dstr}', bbox_inches='tight')

    if plot_histos:
        print('Plotting histograms...')
        # start up histogram
        fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(16, 8))

        # plot them
        strs = [(glove_vals, 'glove(Mij)', 'b'),
                (np.log(w2v_vals), 'log(w2v(Mij))', 'g')]

        for axidx, (vals, xlab, color) in enumerate(strs):
            n_, bins_, patches_ = axs[axidx].hist(vals, 50, density=True,
                                                  facecolor=color, alpha=0.75)
            axs[axidx].set_ylabel('Density')
            axs[axidx].set_xlabel(xlab)
            axs[axidx].grid(True)

        fig.savefig(f'png/PROB_TEST{dstr}', bbox_inches='tight')


##########
def main():
    histogram('/home/rldata/hilbert-embeddings/cooccurrence/5w-dynamic-10k/thresh1',
              True, True, False)

if __name__ == '__main__':
    main()

