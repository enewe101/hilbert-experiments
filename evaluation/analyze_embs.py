import hilbert as h
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rmsd import kabsch
from matplotlib import pyplot as plt
from matplotlib import cm
import sys

path = sys.argv[1]
embs = h.embeddings.Embeddings.load(path, device='cpu')

def most_similar(embs):
    words = ['vector', 'take', 'york', 'the', 'disposable']#['cat', 'hot', 'money', 'cuba']
    
    for w in words:
        print(f'{w}:')
        for b in [True, False]:
            print('{}using covecs'.format('not ' if not b else ''))
            sim_dot = embs.greatest_product(w, covecs=b)[:10]
            sim_cos = embs.greatest_cosine(w, covecs=b)[:10]
            print('\tdot product: {}'.format(', '.join(sim_dot)))
            print('\tcosine sim:  {}'.format(', '.join(sim_cos)))


# first test, compare related and unrelated words
def test_embs(embs):
    simwords = ['money', 'dollar', 'stock', 'buy', 'spend', 'purchase', 'rich']
    difwords = ['dancing', 'cat', 'computer', 'pen', 'listen', 'air', 'take']

    embs.V = torch.FloatTensor(PCA(25).fit_transform(embs.V))

    ssum = embs[simwords[0]]
    for w in simwords[1:]:
        ssum += embs[w]
    ssum /= len(simwords)

    dsum = embs[difwords[0]]
    for w in difwords[1:]:
        dsum += embs[w]
    dsum /= len(difwords)

    colors = ['b', 'r']
    fig, axs = plt.subplots(2, 1, tight_layout=False, figsize=(12, 6))
    x = list(range(len(ssum)))
    for i, vals in enumerate([ssum, dsum]):
        axs[i].bar(x, vals, width=0.9, color=colors[i])

    plt.show()

def hal_examine(embs):
    pca = PCA(2)
    embs.V = torch.FloatTensor(pca.fit_transform(embs.V))
    print('PCA to 25-d explains {:0.4f} percent of variance'.format(
        sum(pca.explained_variance_ratio_)))
    
    words = ['coffee', 'tea', 'road', 'street', 'drive', 'drove']
    rows = np.vstack([embs[w].cpu().numpy() for w in words])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(rows, cmap=cm.bwr)
    ax.set_xticks(np.arange(len(rows[0])))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(words)

    cbar = fig.colorbar(im, orientation='horizontal', pad=0.05, aspect=50)
    fig.tight_layout()
    plt.show()


def viz_embs(name, embs, dopca=True):
    pca = PCA(25) if dopca else TSNE(2)
    try:
        combined = np.vstack([embs.V.numpy(), embs.W.numpy()])
        print(combined.shape)
    except Exception:
        import pdb; pdb.set_trace()
    print('fitting pca...')
    if not dopca:
        idx = np.random.choice(len(embs.V), size=1000, replace=False)
        combined = np.vstack([combined[idx], combined[idx + len(embs.V)]])
        combined = PCA(50).fit_transform(combined)
        print('did first compression')

    data = pca.fit_transform(combined)
    v2 = data[: int(len(data) / 2)]
    w2 = data[int(len(data) / 2):]
    
    # randomly sample subset of 1000 vectors
    idx = np.random.choice(len(v2), size=1000, replace=False)

    plt.figure(figsize=(12, 6))
    plt.plot(v2[idx][:,0], v2[idx][:,1], 'b.', label='term vectors')
    plt.plot(w2[idx][:,0], w2[idx][:,1], 'r.', label='context vectors')
    plt.legend()
    plt.savefig(f'png/{name}-viz', bbox_inches='tight')

#hal_examine(embs)
#test_embs(embs)
most_similar(embs)
#viz_embs(path.split('/')[-1], embs, dopca=False)
import pdb;pdb.set_trace()
exit(0)

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

