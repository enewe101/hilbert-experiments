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


# producers
def get_distr_fun(distr_str):
    if distr_str == 'softmax':
        return lambda tensor: torch.softmax(tensor, dim=1)
    if distr_str == 'sigmoid':
        return torch.sigmoid
    raise NotImplementedError('No distribution function \"{}\"!'.format(distr_str))


def get_act_fun(act_str):
    if act_str == 'sigmoid':
        return torch.sigmoid
    if act_str == 'relu':
        return torch.relu
    if act_str == 'tanh':
        return torch.tanh
    raise NotImplementedError('No activation function \"{}\"!'.format(act_str))



# MLP classifier, as used by Conneau et al. 2017 EMNLP
class MLPClassifier(nn.Module):

    def __init__(self, in_features, n_classes, h_dim=512, dropout=0., nonlinear=False):
        super(MLPClassifier, self).__init__()

        if nonlinear:
            self.model = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, h_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(h_dim, n_classes))
        else:
            self.model = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, n_classes))

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
        return self.embeddings(token_seqs), None


    def get_padding_vec(self):
        return self.embeddings(self.torch_padding_id).reshape(self.emb_dim)
