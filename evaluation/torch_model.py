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
        self.torch_padding_id = torch.LongTensor([self.padding_id]).to(DEVICE)
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
            pads.append(max_len - len(tok_ids)) # number of paddings appended

        # now convert to long tensors
        torch_ids = torch.LongTensor(ids).to(DEVICE)
        torch_pads = torch.LongTensor(pads).to(DEVICE)

        # now finally yield the sequence of embeddings
        return self.embeddings(torch_ids), torch_pads


    def get_padding_vec(self):
        return self.embeddings(self.torch_padding_id).reshape(self.dim)



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
        # shape of embs is batch_size X max_seq_length X embedding_dim
        mb_size, max_len, emb_dim = embs.shape

        mean_seqs = None # mb_size X emb_dim
        if self.do_mean:
            # here we are constructing the padding offsets to factor in during the
            # computation of the mean vectors. Much faster than a for-loop.
            padding_offset = self.get_padding_vec().repeat(mb_size).reshape(mb_size, emb_dim)
            padding_offset *= pads.float().reshape(mb_size, 1)

            # now we sum up the sequences (which includes the padding in them)
            mean_seqs = embs.sum(dim=1)

            # now we subtract away the padding vectors
            mean_seqs -= padding_offset

            # now we divide by the true length of the sequences, giving us the means
            mean_seqs /= (max_len - pads).float().reshape(mb_size, 1)

        max_seqs = None # mb_size X emb_dim
        if self.do_max:
            # here we will have to ignore the padding, it won't have the max values anyway
            max_seqs, _ = embs.max(dim=1)

        # return the big boys
        if (mean_seqs is not None) and (max_seqs is not None):
            return torch.cat((max_seqs, mean_seqs), dim=1)
        return mean_seqs if mean_seqs is not None else max_seqs



# classes for the actual learning models
class LogisticRegression(EmbeddingPooler):
    def __init__(self, h_embs, n_classes,
                 use_vectors=True,
                 pooling='mean'
                 ):
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
    def __init__(self, h_embs, n_classes, hdim1, hdim2,
                 dropout=0.,
                 use_vectors=True,
                 pooling='mean',
                 ):
        super(FFNN, self).__init__(
            h_embs, use_vectors=use_vectors, pooling=pooling
        )
        assert hdim1 > 0 and hdim2 > 0
        # number of input features
        in_features = 2 * self.dim if pooling == 'both' else self.dim
        self.model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hdim1, hdim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hdim2, n_classes)
        )

    def forward(self, token_minibatch):
        pooled_embs = super(FFNN, self).forward(
            token_minibatch,
        )
        return self.model(pooled_embs)
