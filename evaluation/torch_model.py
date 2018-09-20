import numpy as np
import torch
import torch.nn as nn
from hilbert_device import DEVICE

class EmbeddingModel(nn.Module):
    """
    Generic module that stores pre-trained embeddings for any
    other neural model we'll be implementing.
    """
    def __init__(self, h_embs, use_vectors=True):
        super(EmbeddingModel, self).__init__()

        if not use_vectors:
            raise NotImplementedError('Use of vectors & covectors not implemented!')

        _dim = len(h_embs[0])
        _n_embs = len(h_embs.dictionary) + 2 # including unk & padding

        # set ids for the two
        self.unk_id = len(h_embs.dictionary)
        self.padding_id = len(h_embs.dictionary) + 1

        # set up the padding embeddings
        _padding = torch.from_numpy(np.random.normal(
            -0.15, 0.15, _dim
        )).reshape(1, -1).float().to(DEVICE)

        # combine them all
        _all_embs = torch.cat((h_embs.V.float(),
                               h_embs.unk.float().reshape(1, -1),
                               _padding),
                              dim=0).to(DEVICE)

        # now, put the pretrained ones into them
        self.embeddings = nn.Embedding(_n_embs, _dim).to(DEVICE)
        self.embeddings.weight = nn.Parameter(_all_embs, requires_grad=False)
        self.dim = _dim


    def forward(self, sorted_tok_ids):
        """
        :param sorted_tok_ids: sorted by length list of lists of tokens
        :return: tensor of embeddings, with padding, for a downstream model
        """
        max_len = len(sorted_tok_ids[-1]) # must be sorted by length on input
        ids = []
        pads = [] # store indexes when padding starts
        for tok_ids in sorted_tok_ids:

            # add the padding and add the ids
            ids.append(tok_ids + [self.padding_id] * (max_len - len(tok_ids)))
            pads.append(len(tok_ids))

        # now convert to long tensors
        torch_ids = torch.LongTensor(ids).to(DEVICE)
        torch_pads = torch.LongTensor(pads).to(DEVICE)

        # now finally yield the sequence of embeddings
        return self.embeddings(torch_ids), torch_pads



# embedding pooler, useful for FFNNs and Logistic Regression
class EmbeddingPooler(EmbeddingModel):
    """
    Generic class for doing pooled word embeddings. Works quite simply
    - it pools together word embeddings according to either mean
    pooling, max pooling, or both. It ignores padding.
    """
    def __init__(self,
                 h_embs,
                 use_vectors=True,
                 pooling='mean',
                 ):
        super(EmbeddingPooler, self).__init__(h_embs, use_vectors)
        self.pooling = pooling
        self.do_mean = self.pooling == 'both' or self.pooling == 'mean'
        self.do_max = self.pooling == 'both' or self.pooling == 'max'


    def forward(self, token_minibatch):
        embs, pads = super(EmbeddingPooler, self).forward(
            token_minibatch,
        )

        # "extract" the features
        pools = []
        for i, emb_seq in enumerate(embs):
            vec = []
            if self.do_mean:
                vec.append(torch.mean(emb_seq[:pads[i]], dim=0))
            if self.do_max:
                vec.append(torch.max(emb_seq[:pads[i]], dim=0)[0])
            pools.append(torch.cat(vec).reshape(1, -1))

        # now we have all the pools, and we don't include the padding when
        # doing the pooling.
        pools = torch.cat(pools, dim=0)
        return pools



# classes for the actual learning models
class LogisticRegression(EmbeddingPooler):
    def __init__(self, h_embs, n_classes, use_vectors=True, pooling='both'):
        super(LogisticRegression, self).__init__(
            h_embs, use_vectors=use_vectors, pooling=pooling
        )
        # number of input features
        in_features = 2 * self.dim if pooling == 'both' else self.dim
        self.output = nn.Linear(in_features, n_classes)


    def forward(self, token_minibatch):
        pooled_embs = super(LogisticRegression, self).forward(
            token_minibatch,
        )
        return self.output(pooled_embs)


class FFNN(EmbeddingPooler):
    def __init__(self, h_embs, n_classes, hdim1, hdim2, use_vectors=True, pooling='both'):
        super(FFNN, self).__init__(
            h_embs, use_vectors=use_vectors, pooling=pooling
        )
        assert hdim1 > 0 and hdim2 > 0
        # number of input features
        in_features = 2 * self.dim if pooling == 'both' else self.dim
        self.model = nn.Sequential(
            nn.Linear(in_features, hdim1),
            nn.ReLU(),
            nn.Linear(hdim1, hdim2),
            nn.ReLU(),
            nn.Linear(hdim2, n_classes)
        )

    def forward(self, token_minibatch):
        pooled_embs = super(FFNN, self).forward(
            token_minibatch,
        )
        return self.model(pooled_embs)
