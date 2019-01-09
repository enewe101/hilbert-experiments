import sys
from matplotlib import pyplot as plt
import hilbert as h
import shared

PATH = "/home/ndg/projects/shared_datasets/hilbert/embeddings/hbt-svd-s1-tc1e5-v10k-I500/trace.txt"

def plot_trace(path=PATH, display=True):

    X, Y = read_trace(path)

    if display:
        plt.plot(X,Y)
        plt.show()

    return X, Y


def read_trace(path):

    X, Y = [], []
    with open(path) as trace_file:
        for line in trace_file:
            if not line.startswith("iter"):
                continue
            try:
                _, iter_num, _, loss = line.strip().split("\t")
            except ValueError:
                print(line)
                raise
            X.append(int(iter_num))
            Y.append(float(loss))

    return X, Y

if __name__ == '__main__':
    plot_trace(sys.argv[1])
