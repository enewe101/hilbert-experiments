import os
import numpy as np
import argparse
from collections import defaultdict
from sys import argv
from matplotlib import pyplot as plt
from evaluation.dataset_load import HilbertDataset
from evaluation.run_experiments import similarity_exp, load_embeddings

COLORS = ['xkcd:aqua', 'xkcd:blue', 'xkcd:green', 'xkcd:black',
          'xkcd:grey', 'xkcd:magenta', 'xkcd:purple', 'xkcd:red',
          'xkcd:lightblue', 'xkcd:darkblue', 'xkcd:gold', 'xkcd:lime']


def get_all_iters(path):
    iters = []
    for d in os.listdir(path):
        if os.path.isdir(f'{path}/{d}'):
            prefix, iter_ = d.split('-')
            iters.append((f'{path}/{d}', int(iter_)))
    iters.sort(key=lambda t: t[1])
    return iters


def evaluate_embs(path, dataset):
    try:
        return load_emb_results(path, dataset.name)
    except FileNotFoundError:
        pass

    if dataset.name == 'similarity':
        embs = load_embeddings(path)
        results = similarity_exp(embs, dataset, None)
    else:
        raise NotImplementedError(f'Lazy, did not implement results for \"{dataset.name}\"!')

    print(f'{path} - running similarity exp...')
    np.save(f'{path}/{dataset.name}', np.array([results]))
    return results


def load_emb_results(path, dsname):
    for fname in os.listdir(path):
        if dsname in fname:
            return np.load(f'{path}/{fname}')[0]
    try:
        for fname in os.listdir(f'{path}/vectors-final'):
            if dsname in fname:
                return np.load(f'{path}/vectors-final/{fname}')[0]
    except FileNotFoundError:
        pass
    raise FileNotFoundError(f'No results for {path} computed yet!')


def plot_performance(results_list, epochs, targs=None, reskey='full-spearman'):
    results = defaultdict(lambda: [])
    for r in results_list:
        for dsname, dsres in r.results_by_dataset.items():
            results[dsname].append(dsres[reskey])

    plt.figure(figsize=(20, 12))
    ax = plt.subplot(111)

    for i, (dsname, reslist) in enumerate(results.items()):
        ax.plot(epochs, reslist, color=COLORS[i], label=dsname)

        if targs is not None:
            targres = targs.results_by_dataset[dsname][reskey]
            ax.plot([epochs[-1]], [targres], '*', color=COLORS[i], markersize=8)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.set_ylabel('Spearman')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=3, fancybox=True, shadow=True)
    plt.show()


def compare_finals(final_res_dict, target_res, reskey='full-spearman'):
    plt.figure(figsize=(24, 16))
    ax = plt.subplot(111)
    n = len(final_res_dict)
    bar_width = 0.4 / n
    opacity = 0.55

    # one color for each model
    models = list(final_res_dict.keys())

    # now get the datasets
    datasets = list(final_res_dict[models[0]].results_by_dataset.keys())
    index = np.arange(len(datasets))

    for i, m in enumerate(models):
        mname = m
        pred_res = [final_res_dict[m].results_by_dataset[k][reskey] for k in datasets]
        rects_i = ax.bar(index + (i * bar_width), pred_res, bar_width,
                        alpha=opacity, color=COLORS[i], label=mname)

    if target_res is not None:
        targ_res = [target_res.results_by_dataset[ds][reskey] for ds in datasets]
        rects_t = ax.bar(index + (n * bar_width), targ_res, bar_width,
                        alpha=opacity, color='y', label='target')

    ax.set_xlabel('dataset')
    ax.set_xticks(index + n * bar_width / 2)
    ax.set_xticklabels( [x[:12] for x in datasets], rotation=70)
    ax.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, nargs='+',
                        help='the target director(y/ies) to evaluate embeddings')

    parser.add_argument('-c', '--compareto', type=str, default='',
                        help='the directory containing embeddings we would like to'
                             ' compare the target to')

    parser.add_argument('-e', '--eval', action='store_true',
                        help='set to evaluate all embeddings in target')

    parser.add_argument('--base', type=str, default='/home/rldata/hilbert-embeddings/embeddings',
                        help='base directory containg any embeddings we will be testing')

    args = parser.parse_args()
    analyze_iters = len(args.target) == 1

    # get all of the iterations
    iter_files = None
    if analyze_iters:
        print('One target directory specified, analyzing its iterations...')
        iter_files = get_all_iters(f'{args.base}/{args.target[0]}')
        print(', will be processing {} files...'.format(len(iter_files)))
    else:
        print('Multiple target directories, analyzing final results...')

    # get the last files of the iters in each of the targets
    final_files = {}
    for d in args.target:
        files = get_all_iters(f'{args.base}/{d}') if iter_files is None else iter_files
        final_files[d] = files[-1][0] # just the file name, not epoch

    # may need to do the experiments on them
    if args.eval:
        print('Loading datasets for running tasks...')
        h_datasets = np.load('np/all_data.npz')['arr_0'][0]
        simds = h_datasets['similarity']
        eval_fun_arg = (evaluate_embs, simds)
    else:
        eval_fun_arg = (load_emb_results, 'similarity')

    # do the evaluation function & do the analysis after
    target_results = None
    if args.compareto:
        targpath = f'{args.base}/{args.compareto}'
        target_results = eval_fun_arg[0](targpath, eval_fun_arg[1])

    if iter_files:
        iter_epochs = []
        iter_results = []
        for path, epoch in iter_files:
            iter_results.append(eval_fun_arg[0](path, eval_fun_arg[1]))
            iter_epochs.append(epoch)
        plot_performance(iter_results, iter_epochs, targs=target_results)
        final_results = {args.target[0]: iter_results[-1]}
    else:
        final_results = {}
        for model, path in final_files.items():
            final_results[model] = eval_fun_arg[0](path, eval_fun_arg[1])

    compare_finals(final_results, target_results)



if __name__ == '__main__':
    main()
