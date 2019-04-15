import torch

from evaluation.run_experiments import load_embeddings
from evaluation.torch_model import BasicAttention, NeuralAttention



if __name__ == '__main__':

    h_embs = load_embeddings('/Users/kiankd/git/hilbert-experiments/test_embs/HBT-MLE-FINAL-v50k', device='cpu')
    for b in [True, False]:
        for c in [BasicAttention, NeuralAttention]:
            kwargs = {'learn_W': b} if c==BasicAttention else {}
            model = c(h_embs, 2, dropout=0.5, **kwargs)

            # X is a list of lists; i.e., a batch of indices with 0 padding
            X = torch.LongTensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 123, 44],
                    [10, 20, 30, 40, 50, 60, 70, 80, 90, 3, 0],
                    [2, 4, 6, 8, 10, 12, 14, 16, 18, 12, 0],
                    [1000, 1282, 3321, 4123, 5123, 6342, 7551, 8331, 4329, 0, 0],
                    [9132, 2241, 3234, 4511, 5234, 56, 7222, 8111, 9324, 0, 0],
                    [6523, 2141, 3294, 4411, 523, 561, 4222, 1111, 9323, 0, 0],
                ]
            )
            pads = torch.LongTensor([0, 1, 1, 2, 2, 2])
            Y = torch.LongTensor([0,0,0,1,1,1]) # 6 labels

            preds = model(X, pads=pads)


