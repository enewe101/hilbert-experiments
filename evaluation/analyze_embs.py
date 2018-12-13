import hilbert as h
import numpy as np
import torch
from rmsd import kabsch
from matplotlib import pyplot as plt

path = '/home/rldata/hilbert-embeddings/embeddings/std-w2v-s1-t2e5-v10k-iter5/vectors-final'
embs = h.embeddings.Embeddings.load(path, device='cpu')

# first test, compare related and unrelated words
def test_embs():
    simwords = ['money', 'dollar', 'stock', 'buy', 'spend', 'purchase', 'rich']
    difwords = ['dance', 'cat', 'computer', 'pen', 'listen', 'air', 'take']

    ssum = embs[simwords[0]]
    for w in simwords[1:]:
        ssum += embs[w]
    ssum /= len(simwords)

    dsum = embs[difwords[0]]
    for w in difwords[1:]:
        dsum += embs[w]
    dsum /= len(difwords)

    colors = ['b', 'r']
    fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(16, 10))
    x = list(range(len(ssum)))
    for i, vals in enumerate([ssum, dsum]):
        axs[i].bar(x, vals, width=0.9, color=colors[i])

    plt.show()

# test the rotational similarity
avgv = torch.mean(embs.V, dim=0).reshape(1, -1)
avgw = torch.mean(embs.W, dim=0).reshape(1, -1)
target = torch.zeros(avgv.shape)
target[0] = 1.
target = target.reshape(1, -1)
R = kabsch(avgv, target)

rotated_V = embs.V.numpy() #@ R
rotated_W = embs.W.numpy() #@ R

rR = kabsch(rotated_V, rotated_W)
for i, v in enumerate(rR):
    diag = v[i]
    s = (np.sum(v) - diag) / 299
#    print(diag, s)

avgv = avgv.reshape(-1)
avgw = avgw.reshape(-1)
cos = avgv.dot(avgw) / (avgv.norm() * avgw.norm())
print(cos.item())

