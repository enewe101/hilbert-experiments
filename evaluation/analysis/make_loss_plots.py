from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import os
import shared
from sys import argv
import re
from hilbert_histograms import save_figure


BASE = shared.CONSTANTS.EMBEDDINGS_DIR
print(BASE)


def read_trace(pth):
    losses = []
    with open(pth, 'r') as f:
        for line in f:
            if line.startswith('iter'):
                val = line.split('loss=')[1].rstrip('\n')
                losses.append(float(val))
    return losses


def load_traces(gs_prefix):
    traces = {}
    for f in os.listdir(BASE):
        if f == gs_prefix: #f.startswith(gs_prefix):
            pth = os.path.join(BASE, f)
            try:
                tr = read_trace(os.path.join(pth, 'trace.txt'))
                traces[f] = tr
            except FileNotFoundError:
                pass
    return traces


def filter_by_lr(keys):
    lrd = defaultdict(lambda: [])
    for k in keys:
        # example: auto-hbt-glv-v10k-u1e05-l5e02-sadam
        lr_s = re.search(r'-l[^-]*', k).group(0)
        lr = lr_s.lstrip('-l').replace('e', 'e-').replace('p', '.')
        lr = float(lr)
        
        mb_s = re.search(r'-u[^-]*', k+'-u1').group(0)
        mb = mb_s.lstrip('-u').replace('e', 'e-').replace('p', '.')
        mb = float(mb)

        lrd[lr].append((k, mb, lr,))
    return list(lrd.values())
        

def make_plots(gs_prefix):
    traces = load_traces(gs_prefix)
    
    for comparables in filter_by_lr(traces.keys()):
        fig, axs = plt.subplots(1, 1, figsize=(16,10))
        lr = str(comparables[0][-1])

        axs.set_title(f'Loss comparison for learning rate: {lr}')
        for key, mb, _ in comparables:
            name = f'mb = {mb}'
            loss = traces[key]
            
            if not loss:
                continue

            x = np.arange(len(loss))
            axs.plot(x, loss, '-', label=name)
        axs.set_xlabel('Iterations')
        axs.set_ylabel('Loss')
        axs.legend()
        
        save_figure(fig, '{}-mb-comparison-{}'.format(
            gs_prefix, str(lr).replace('0.', '1e')))


def avgmin(vals, k=10):
    if len(vals) < k:
        return np.inf
    best = np.inf
    for i in range(len(vals) - k):
        crt = np.mean(vals[i:i+k])
        best = min(best, crt)
    return best

def print_bests(gs_prefix):
    traces = load_traces(gs_prefix)
    res = {k: avgmin(v) if v else np.inf for k,v in traces.items()}
    result = list(res.items())
    result.sort(key=lambda t: t[1])
    for key, loss in result:
        print('{:20}: min loss = {:.6f}'.format(key, loss))
    

###################3
def main():
    prefix = argv[1]

    try:
        option = argv[2]
        if option == '1':
            print_bests(prefix)
            exit(0)

    except IndexError:
        pass

    make_plots(prefix)


if __name__ == '__main__':
    main()

