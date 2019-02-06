import argparse
import os
import numpy as np

NAN = 0
EMPTY = 1

def numbercheck(x):
    return type(x) != str


def update_res(old_res, iter_resholder, iterp, rkey='full-spearman'):
    iter_res = {'iter': iterp}
    for k in iter_resholder.results_by_dataset:
        iter_res[k] = iter_resholder.results_by_dataset[k][rkey]
        if np.isnan(iter_res[k]):
            if len(old_res) > 0:
                return old_res
            return NAN

    # this is the first time its called
    if len(old_res) == 0:
        return iter_res
  
    # get the results for the current iteration
    iter_vals = [x for x in iter_res.values() if numbercheck(x)]
    iter_score = np.mean(iter_vals)

    # get the results we had before
    old_vals = [x for x in old_res.values() if numbercheck(x)]
    old_score = np.mean(old_vals)

    # make sure these are results from all 12 similarity datasets
    assert len(old_vals) == len(iter_vals) and len(iter_vals) >= 10

    # now compare and update if necessary
    if iter_score > old_score:
        return iter_res
    return old_res


def get_best(exp_path, check=False):

    # all the iterations
    results = {}

    # get all the iteration directories
    files = [x for x in os.listdir(exp_path) if x.startswith('iter')]

    # this is NAN because it means that the model diverged within the first
    # 1000 epochs, before being able to write even one set of embeddings
    if len(files) == 0:
        return NAN

    for f in sorted(files, key=lambda x: int(x.lower().lstrip('iter-'))):
        iter_path = os.path.join(exp_path, f)
        has_res = any(map(lambda x: x == 'similarity.npy', os.listdir(iter_path)))
        nan_res = any(map(lambda x: x == 'similarity_NAN.npy', os.listdir(iter_path)))

        if nan_res and len(results) == 0:
            return NAN
        elif nan_res:
            return results

        if has_res:
            if check:
                continue
            simres = np.load(os.path.join(iter_path, 'similarity.npy'))[0]
            results = update_res(results, simres, iter_path)
        elif len(results) == 0:
            return EMPTY
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gs', type=str, help='prefix of the gridsearch')
    parser.add_argument('--base', type=str, default='/home/enewel3/scratch/hilbert-data/embeddings')
    parser.add_argument('--todo', action='store_true', help='find empty unanalyzed runs')
    args = parser.parse_args()

    results = {}
    header = []
    for exp in os.listdir(args.base):
        if exp.startswith(args.gs):
            print()
            print('exp',exp)
            path = os.path.join(args.base, exp)
            results[exp] = get_best(path, check=args.todo)
            print('res',results[exp])
            if len(header) == 0 and type(results[exp]) == dict:
                header = list(results[exp].keys())
                if len(header) > 0:
                    header.remove('iter')
                print('header',header)

    if args.todo:
        emptys = [exp for exp in results if results[exp] == EMPTY]
        print('Number of empty runs:', len(emptys))
        with open('/home/enewel3/naacl2019/runs_todo.txt', 'w') as f:
            for e in emptys:
                f.write(f'{e}\n')
        exit(0)

    nans = 0
    with open(f'/home/enewel3/naacl2019/{args.gs}.csv', 'w') as csvf:
        csvf.write('experiment name,{}\n'.format(','.join(header)))
        for exp, res in results.items():
            try:
                if res == EMPTY:
                    print('EMPTY')
                elif res == NAN:
                    print('NAN')
                    nans += 1
                    values = ['NAN']*len(header)
                else:
                    values = [str(res[k]) for k in header]
            except KeyError:
                import pdb; pdb.set_trace()
            csvf.write('{},{}\n'.format(exp, ','.join(values)))
    print('We have {} nans out of {}'.format(nans, len(results)))


if __name__ == '__main__':
    main()


