import numpy as np
import torch
import torch.nn as nn
from evaluation.hparams import HParams



##### Helper functions! #####
def torch_scalar(scalar):
    return torch.FloatTensor([scalar]).to(HParams.DEVICE)


def rmatrix(d):
    return nn.init.xavier_uniform_(torch.zeros((d, d),
                                   device=HParams.DEVICE))

def rvector(d):
    return nn.init.xavier_uniform_(torch.zeros((d, 1),
                                   device=HParams.DEVICE)).reshape(-1)

def _build_padding_mask(B, L, pads):
    """ Returns a BxL matrix for padding! """
    ## we will now construct the padding matrix from the pads LongTensor
    P = torch.empty((L, B), device=HParams.DEVICE)  # will transpose after
    P[:] = L - pads  # now it is a matrix with number of words as each value
    P = P.t().float()  # transpose back to be proper

    # now we are building up a matrix of indices to properly build the padding matrix
    Z = torch.empty((B, L), device=HParams.DEVICE)
    Z[:] = torch.arange(L).float()

    # now use np.where to figure out where to put infinities for padding
    mask = torch.where(Z < P,
                       torch.FloatTensor([1]).to(HParams.DEVICE),
                       torch.FloatTensor([np.inf]).to(HParams.DEVICE))
    return mask


def _mask_to_tensor(mask, bsz):
    mT = torch.bmm(mask.unsqueeze(2), mask.reshape(bsz, 1, -1))
    mT -= 1 # turn the ones into zeros
    mT[mT == np.inf] = -1e6
    return mT


def apply_energy_mask(Es, bsz, max_len, pads):
    mask = _build_padding_mask(bsz, max_len, pads)
    mT = _mask_to_tensor(mask, bsz)
    mEs = Es + mT
    return mEs



# MLP classifier, as used by Conneau et al. 2017 EMNLP
class MLPClassifier(nn.Module):

    def __init__(self, in_features, n_classes, h_dim=512, dropout=0., linear=False):
        super(MLPClassifier, self).__init__()

        if linear:
            self.model = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, n_classes))
        else:
            self.model = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(h_dim, n_classes))

    def forward(self, X):
        return self.model(X)



# Generic model to feed forward token sequences to embeddings
class EmbeddingModel(nn.Module):
    """
    Generic module that stores pre-trained embeddings for any
    other neural model we'll be implementing.
    """
    def __init__(self, h_embs, fine_tune=True, zero_padding=False, store_covecs=False):
        super(EmbeddingModel, self).__init__()

        _dim = h_embs.dim
        _n_embs = len(h_embs.dictionary) + 2 # including unk & padding

        # set ids for the two
        self.unk_id = len(h_embs.dictionary)
        self.padding_id = len(h_embs.dictionary) + 1

        # set up the padding embeddings
        if zero_padding:
            _padding = torch.zeros(1, _dim).to(HParams.DEVICE)

        else:
            _padding = torch.from_numpy(np.random.normal(
                -0.15, 0.15, _dim
            )).reshape(1, -1).float().to(HParams.DEVICE)


        # combine them all
        _all_embs = torch.cat((h_embs.matrix.float(),
                               h_embs.unk.float().reshape(1, -1),
                               _padding), dim=0).to(HParams.DEVICE)

        # now, put the pretrained ones into them
        self.torch_padding_id = torch.LongTensor([self.padding_id]).to(HParams.DEVICE)

        # if we want to use zero padding we need this kwarg
        _emb_kwarg = {'padding_idx': self.padding_id} if zero_padding else {}
        self.embeddings = nn.Embedding(_n_embs, _dim, **_emb_kwarg)
        self.embeddings.weight = nn.Parameter(_all_embs, requires_grad=fine_tune)
        self.emb_dim = _dim

        # store the covectors in an nn.Embeddings?
        if store_covecs:
            _covecs = torch.cat((h_embs.covecs.float(),
                                 h_embs.unk.float().reshape(1, -1),
                                 _padding), dim=0).to(HParams.DEVICE)
            self.covec_embeddings = nn.Embedding(_n_embs, _dim, **_emb_kwarg)
            self.embeddings.weight = nn.Parameter(_covecs, requires_grad=fine_tune)


    def forward(self, token_seqs, get_covecs=False):
        if get_covecs:
            return self.embeddings(token_seqs), self.covec_embeddings(token_seqs)
        return self.embeddings(token_seqs)


    def get_padding_vec(self):
        return self.embeddings(self.torch_padding_id).reshape(self.emb_dim)



# embedding pooler, useful for FFNNs and Logistic Regression
class EmbeddingPooler(EmbeddingModel):
    """
    Generic class for doing pooled word embeddings. Works quite simply
    - it pools together word embeddings according to either mean
    pooling, max pooling, or both. It appropriately ignores padding.
    """
    def __init__(self, h_embs, pooling='mean', **kwargs):
        super(EmbeddingPooler, self).__init__(h_embs, **kwargs)
        self.pooling = pooling
        self.do_mean = self.pooling == 'both' or self.pooling == 'mean'
        self.do_max = self.pooling == 'both' or self.pooling == 'max'


    def forward(self, token_seqs, pads=None):
        embs = super(EmbeddingPooler, self).forward(token_seqs)

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

