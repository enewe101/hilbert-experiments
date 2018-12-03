import argparse
import os
import numpy as np

NAN = 0
EMPTY = 1


def update_res(resd, resholder, iterp, rkey='full-spearman'):
    old_score = -10. if len(resd)==0 else np.mean([x for x in resd.values() if type(x) == float])
    new_score = np.mean([d[rkey] for d in resholder.results_by_dataset.values()])
    
    if new_score <= old_score:
        return resd

    new_res = {'iter': iterp}
    for k in resholder.results_by_dataset:
        new_res[k] = resholder.results_by_dataset[k][rkey]
    return new_res


def get_best(exp_path):

    # all the iterations
    results = {}

    for f in sorted([x for x in os.listdir(exp_path) if x.startswith('iter')], key=lambda x: int(x.lower().lstrip('iter-'))):
        iter_path = os.path.join(exp_path, f)
        has_res = any(map(lambda x: x == 'similarity.npy', os.listdir(iter_path)))
        nan_res = any(map(lambda x: x == 'similarity_NAN.npy', os.listdir(iter_path)))

        if nan_res and len(results) == 0:
            return NAN
        elif nan_res:
            return results

        if has_res:
            results = update_res(results, np.load(os.path.join(iter_path, 'similarity.npy'))[0], iter_path)
        elif len(results) == 0:
            return EMPTY

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gs', type=str, help='prefix of the gridsearch')
    parser.add_argument('--base', type=str, default='/home/kiankd/scratch/hilbert-data/embeddings')
    args = parser.parse_args()

    results = {}
    header = []
    for exp in os.listdir(args.base):
        if exp.startswith(args.gs):
            path = os.path.join(args.base, exp)
            results[exp] = get_best(path)
            if len(header) == 0 and type(results[exp]) == dict:
                header = list(results[exp].keys())

    emptys = [exp for exp in results if results[exp] == EMPTY]
    print('Number of empty runs:', len(emptys))
#    with open('/home/kiankd/naacl2019/runs_todo.txt', 'w') as f:
#        for e in emptys:
#            f.write(f'{e}\n')

    nans = 0
    with open(f'/home/kiankd/naacl2019/{args.gs}.csv', 'w') as csvf:
        csvf.write('experiment name,{}\n'.format(','.join(header)))
        for exp, res in results.items():
            if res == EMPTY:
                continue
            elif res == NAN:
                nans += 1
                values = ['NAN']*len(header)
            else:
                values = [str(res[k]) for k in header]
            csvf.write('{},{}\n'.format(exp, ','.join(values)))
    print('We have {} nans out of {}'.format(nans, len(results)))


if __name__ == '__main__':
    main()


