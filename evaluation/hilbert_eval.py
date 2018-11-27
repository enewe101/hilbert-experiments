import os
import numpy as np
import argparse
from collections import defaultdict
from matplotlib import pyplot as plt
from evaluation.dataset_load import HilbertDataset
from evaluation.run_experiments import similarity_exp, load_embeddings

COLORS = ['xkcd:aqua', 'xkcd:blue', 'xkcd:green', 'xkcd:black',
          'xkcd:grey', 'xkcd:magenta', 'xkcd:purple', 'xkcd:red',
          'xkcd:lightblue', 'xkcd:darkblue', 'xkcd:gold', 'xkcd:lime']


def get_all_iters(path):
    iters = []
    for i, d in enumerate(reversed(sorted(os.listdir(path)))):
        if os.path.isdir(f'{path}/{d}'):
            try:
                prefix, iter_ = d.split('-')
                iters.append((f'{path}/{d}', int(iter_)))
            except ValueError:
                iters.append((f'{path}/{d}', i))
    iters.sort(key=lambda t: t[1])
    return iters


def evaluate_embs(path, dataset, avg_vw=False):
    try:
        return load_emb_results(path, dataset.name)
    except FileNotFoundError:
        pass

    if dataset.name == 'similarity':
        embs = load_embeddings(path)
        results = similarity_exp(embs, dataset, None, avg_vw=avg_vw)
    else:
        raise NotImplementedError(f'Lazy, did not implement results for \"{dataset.name}\"!')

    print(f'{path} - running similarity exp...')
    np.save(f'{path}/{dataset.name}', np.array([results]))
    return results


def load_emb_results(path, dsname):
    for fname in os.listdir(path):
        if fname == f'{dsname}.npy':
            return np.load(f'{path}/{fname}')[0]
    try:
        for fname in os.listdir(f'{path}/vectors-final'):
            if fname == f'{dsname}.npy':
                return np.load(f'{path}/vectors-final/{fname}')[0]
    except FileNotFoundError:
        pass
    raise FileNotFoundError(f'No results for {path} computed yet!')


def plot_performance(results_list, epochs, targs=None, reskey='full-spearman', save=False):
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

    if save:
        pass
    else:
        plt.show()


def compare_finals(final_res_dict, target_res, reskey='full-spearman', save=False):
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

    # to compares
    res_to_compare = {}

    for i, m in enumerate(models):
        mname = m
        pred_res = [final_res_dict[m].results_by_dataset[ds][reskey] for ds in datasets]
        rects_i = ax.bar(index + (i * bar_width), pred_res, bar_width,
                         alpha=opacity, color=COLORS[i], label=mname)
        res_to_compare[m] = pred_res

    if target_res is not None:
        targ_res = [target_res.results_by_dataset[ds][reskey] for ds in datasets]
        rects_t = ax.bar(index + (n * bar_width), targ_res, bar_width,
                        alpha=opacity, color='y', label='target')

        # now do comparison
        for m, pres in res_to_compare.items():
            p = np.array(pres)
            t = np.array(targ_res)
            dev = 100 * np.mean(p - t)
            print('\nResults for {}: avg. {:2.3f}% absolute difference from targets.'.format(m, dev))

    ax.set_xlabel('dataset')
    ax.set_xticks(index + n * bar_width / 2)
    ax.set_xticklabels( [x[:12] for x in datasets], rotation=70)
    ax.legend()

    if save:
        pass
    else:
        plt.show()


def compare_and_save(sample_res, target_res, save_path, reskey='full-spearman'):
    errors_by_ds = {}

    for ds, ds_res in sample_res.results_by_dataset.items():
        ds_targ_res = target_res.results_by_dataset[ds]
        approx = ds_res[reskey]
        actual = ds_targ_res[reskey]
        errors_by_ds[ds] = (approx - actual) / actual # relative difference

    errors_by_ds['mean'] = np.mean(list(errors_by_ds.values()))
    print('Average relative difference between \"{}\" and target: '
          '{:2.2f}%'.format(save_path, 100 * errors_by_ds['mean']))

    np.save(f'{save_path}/similarity_comparison', np.array([errors_by_ds]))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, nargs='+',
                        help='the sample director(y/ies) to evaluate embeddings')

    parser.add_argument('-c', '--compareto', type=str, default='',
                        help='the directory containing embeddings we would like to'
                             ' compare the target to')

    parser.add_argument('-C', '--save_compare', action='store_true',
                        help='serialize the comparison results into the results file')

    parser.add_argument('-e', '--eval', action='store_true',
                        help='set to evaluate all embeddings in target')

    parser.add_argument('-v', '--viz', action='store_true',
                        help='set to show performance charts')

    parser.add_argument('-s', '--save', action='store_true',
                        help='set to save performance charts and not show')

    parser.add_argument('-G', '--glove_avg', action='store_true',
                        help='set to average the vectors and covectors in experiments')

    parser.add_argument('--base', type=str, default='/home/rldata/hilbert-embeddings/embeddings',
                        help='base directory containg any embeddings we will be testing')

    args = parser.parse_args()
    analyze_iters = len(args.sample) == 1

    # do an arg check
    if 'glv' in args.sample[0] and not args.glove_avg:
        print('\n\n-----------WARNING----------')
        print('You are NOT averaging vectors and covectors, but you ARE using Glove!')
        print('You should probably be passing the -G argument in this case!\n\n')

    # get all of the iterations
    iter_files = None
    if analyze_iters:
        print('One target directory specified, analyzing its iterations...')
        iter_files = get_all_iters(f'{args.base}/{args.sample[0]}')
        print(', will be processing {} files...'.format(len(iter_files)))
    else:
        print('Multiple target directories, analyzing final results...')

    # get the last files of the iters in each of the targets
    final_files = {}
    for d in args.sample:
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

    # here we need to apply the evaluation function to each iteration in the sample directory
    if args.eval:
        eval_fun = lambda p, ds: evaluate_embs(p, ds, avg_vw=args.glove_avg)
        eval_fun_arg = (eval_fun, simds)

    # get all results for eat iteration
    if iter_files:
        iter_epochs = []
        iter_results = []
        comparison_results = {}
        for path, epoch in iter_files:
            iter_results.append(eval_fun_arg[0](path, eval_fun_arg[1]))
            iter_epochs.append(epoch)
            comparison_results[path] = iter_results[-1]

        if args.viz or args.save:
            plot_performance(iter_results, iter_epochs, targs=target_results, save=args.save)

        final_results = {args.sample[0]: iter_results[-1]}
    else:
        final_results = {}
        comparison_results = {}
        for model, path in final_files.items():
            final_results[model] = eval_fun_arg[0](path, eval_fun_arg[1])
            comparison_results[path] = final_results[model]

    # this plots and compares final results
    if args.viz or args.save:
        compare_finals(final_results, target_results, save=args.save)

    # need to compare final to targets and save them
    if args.save_compare and args.compareto:
        for path, f_res in comparison_results.items():
            compare_and_save(f_res, target_results, f'{path}')



# TODO: SVD comparison, determine how the decomposition works

if __name__ == '__main__':
    main()
