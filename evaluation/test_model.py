import torch
import numpy as np
from evaluation.run_experiments import load_embeddings
from evaluation.torch_model_base import _build_padding_mask, _mask_to_tensor
from evaluation.torch_model import BasicAttention, NeuralAttention, BiLSTMClassifier, BasicPooling



if __name__ == '__main__':
    # X is a list of lists; i.e., a batch of indices with 0 padding
    padding_id = 50000 + 1

    X = torch.LongTensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 123, 44],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 3, padding_id],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 12, padding_id],
            [1000, 1282, 3321, 4123, 5123, 6342, 7551, 8331, 4329, padding_id, padding_id],
            [9132, 2241, 3234, 4511, 5234, 56, 7222, 8111, 9324, padding_id, padding_id],
            [6523, 2141, 3294, 4411, 523, 561, 4222, 1111, 9323, padding_id, padding_id],
        ]
    )
    pads = torch.LongTensor([0, 1, 1, 2, 2, 2])
    Y = torch.LongTensor([0, 0, 0, 1, 1, 1])  # 6 labels

    B, L = len(X), len(X[0])

    # padding testing
    mask = _build_padding_mask(B, L, pads)
    assert (mask.shape == (B, L))

    for m, x in zip(mask, X):
        for mij, xij in zip(m, x):
            assert (mij == np.inf and xij == padding_id) or \
                   (mij != np.inf and xij != padding_id)

    mT = _mask_to_tensor(mask, B)
    pad_energy = -1e6

    assert (mT.shape == (B, L, L))
    for s, energy_mask in enumerate(mT):
        assert (energy_mask.shape == (L, L))
        if pads[s] > 0:
            for j in range(L):
                for i in range(L):
                    if i >= L - pads[s] or j >= L - pads[s]:
                        assert (energy_mask[i,j] == pad_energy)
                    else:
                        assert (energy_mask[i,j] != pad_energy)
        else:
            assert (torch.all(energy_mask == torch.zeros((L, L))))


    h_embs = load_embeddings('additional/HBT-MLE-FINAL-v50k', device='cpu',
                             normalize=True, standardize=True)

    # LSTM testing
    for b in [True, False]:
        lstm = BiLSTMClassifier(h_embs, 2, 128, n_layers=1, max_pool=b)
        preds = lstm(X, pads)

        basic = BasicPooling(h_embs, 2, pooling='mean', usecovecs=b)
        bpreds = basic(X, pads)

        for c in [BasicAttention, NeuralAttention]:
            for ffnn in [True, False]:
                for usecovecs in [True, False]:
                    kwargs = {'learn_W': b} if c==BasicAttention else {}
                    model = c(h_embs, 2, dropout=0.5, ffnn=ffnn, usecovecs=usecovecs,
                              **kwargs)
                    preds = model(X, pads=pads)


