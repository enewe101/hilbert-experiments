import matplotlib
import hilbert as h
import shared

PATH = "/home/ndg/projects/shared_datasets/hilbert/embeddings/hbt-svd-s1-tc1e5-v10k-I500/trace.txt"

def plot_trace(path=PATH, display=True):

    X, Y = read_trace(path)

    if display:
        plt.plot(X,Y)
        plt.show

    return X, Y


def read_trace(path):

    X, Y = [], []
    with open(path) as trace_file:
        for line in trace_file:
            if not line.startswith(""):
                continue
            _, iter_num, _, loss = line.strip().split("\t")
            Y.append(int(iter_num))
            X.append(float(loss))

    return X, Y
