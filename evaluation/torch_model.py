import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluation.hparams import HParams


# Generic model to feed forward token sequences to embeddings
# TODO: allow for learning the embeddings from scratch.
class EmbeddingModel(nn.Module):
    """
    Generic module that stores pre-trained embeddings for any
    other neural model we'll be implementing.
    """
    def __init__(self, h_embs, use_vectors=True, zero_padding=False):
        super(EmbeddingModel, self).__init__()

        if not use_vectors:
            raise NotImplementedError('Use of vectors & covectors not implemented!')

        _dim = len(h_embs[0])
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
        _all_embs = torch.cat((h_embs.V.float(),
                               h_embs.unk.float().reshape(1, -1),
                               _padding), dim=0).to(HParams.DEVICE)

        # now, put the pretrained ones into them
        self.torch_padding_id = torch.LongTensor([self.padding_id]).to(HParams.DEVICE)

        # if we want to use zero padding we need this kwarg
        _emb_kwarg = {'padding_idx': self.padding_id} if zero_padding else {}
        self.embeddings = nn.Embedding(_n_embs, _dim, **_emb_kwarg).to(HParams.DEVICE)
        self.embeddings.weight = nn.Parameter(_all_embs, requires_grad=True)
        self.emb_dim = _dim


    def forward(self, sorted_tok_ids):
        """
        :param sorted_tok_ids: sorted by length list of lists of tokens
        :return: tensor of embeddings, with padding, for a downstream model
        """
        # must be sorted by length on input, either increasing or decreasing
        max_len = max(len(sorted_tok_ids[0]), len(sorted_tok_ids[-1]))

        ids = [] # the actual token ids
        pads = [] # store number of pads in each thing
        for tok_ids in sorted_tok_ids:

            # add the padding and add the ids
            ids.append(tok_ids + [self.padding_id] * (max_len - len(tok_ids)))
            pads.append(max_len - len(tok_ids)) # number of paddings appended

        # now convert to long tensors
        torch_ids = torch.LongTensor(ids).to(HParams.DEVICE)
        torch_pads = torch.LongTensor(pads).to(HParams.DEVICE)

        # now finally yield the sequence of embeddings
        return self.embeddings(torch_ids), torch_pads


    def get_padding_vec(self):
        return self.embeddings(self.torch_padding_id).reshape(self.emb_dim)



# embedding pooler, useful for FFNNs and Logistic Regression
class EmbeddingPooler(EmbeddingModel):
    """
    Generic class for doing pooled word embeddings. Works quite simply
    - it pools together word embeddings according to either mean
    pooling, max pooling, or both. It ignores padding.
    """
    def __init__(self, h_embs, use_vectors=True, pooling='mean'):
        super(EmbeddingPooler, self).__init__(h_embs, use_vectors)
        self.pooling = pooling
        self.do_mean = self.pooling == 'both' or self.pooling == 'mean'
        self.do_max = self.pooling == 'both' or self.pooling == 'max'


    def forward(self, token_minibatch):
        embs, pads = super(EmbeddingPooler, self).forward(token_minibatch)

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
    def __init__(self, h_embs, n_classes, use_vectors=True, pooling='mean'):
        super(LogisticRegression, self).__init__(
            h_embs, use_vectors=use_vectors, pooling=pooling)

        # number of input features
        in_features = 2 * self.emb_dim if pooling == 'both' else self.emb_dim
        self.output = nn.Linear(in_features, n_classes)


    def forward(self, token_minibatch):
        pooled_embs = super(LogisticRegression, self).forward(token_minibatch)
        return self.output(pooled_embs)



# Basic FNN for classification on pooled word embeddings
class FFNN(EmbeddingPooler):
    def __init__(self, h_embs, n_classes, hdim1, hdim2,
                 dropout=0.,
                 use_vectors=True,
                 pooling='max'):
        super(FFNN, self).__init__(h_embs, use_vectors=use_vectors, pooling=pooling)
        assert hdim1 > 0 and hdim2 > 0

        # number of input features
        in_features = 2 * self.emb_dim if pooling == 'both' else self.emb_dim
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
        pooled_embs = super(FFNN, self).forward(token_minibatch)
        return self.model(pooled_embs)



# Basic BiLSTM for classification of word sequences.
class BiLSTMClassifier(EmbeddingModel):

    def __init__(self, h_embs, n_labels, rnn_hdim, n_layers=1, use_vectors=True, dropout=0):
        super(SeqLabLSTM, self).__init__(h_embs, use_vectors=use_vectors, zero_padding=True)
        assert rnn_hdim > 0 and n_labels > 0 and n_layers > 0

        self.hidden_dim = rnn_hdim
        self.n_layers = n_layers
        self.n_labels = n_labels

        # the big boy LSTM that does all the work
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # output label prediction at each time step
        self.hidden_to_label = nn.Linear(self.hidden_dim * 2, self.n_labels)

        # don't do hidden initialization until we know the batch size
        self.hidden = None


    def init_hidden(self, mb_size):
        hstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        cstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        return hstate, cstate


    def forward(self, sorted_tok_ids):
        # get the tensor with emb sequences, along with the number of pads in each seq
        emb_seqs, pads = super(SeqLabLSTM, self).forward(sorted_tok_ids)

        # now we gotta do some special packing
        # note: emb_seqs -> (batch_size, max_seq_len, embedding_dim)
        bsz, max_len, emb_dim = emb_seqs.shape
        self.hidden = self.init_hidden(bsz)
        X = nn.utils.rnn.pack_padded_sequence(emb_seqs, max_len - pads, batch_first=True)

        # feed throught the bilstm
        X, (hidden_state, cell_state) = self.lstm(X, self.hidden)
        last_backward = hidden_state[-1, :, :]
        last_forward = hidden_state[-2, :, :]
        state_concat = torch.cat((last_forward, last_backward), dim=1)

        # run through the linear tag prediction
        Y = self.hidden_to_label(state_concat) # dim is bsz X n_labels

        # softmax activations in the feed forward for an easy main method
        Y_hat = F.log_softmax(Y, dim=1)
        return Y_hat.view(bsz, self.n_labels)



# LSTM for sequence labelling (POS-tagging!)
class SeqLabLSTM(EmbeddingModel):

    # universal constant, do not change!
    PADDING_LABEL_ID = 0


    # extends the EmbeddingModel class which uses our pretrained embeddings.
    def __init__(self, h_embs, n_labels, rnn_hdim,
                 n_layers=1,
                 use_vectors=True,
                 bidirectional=True,
                 dropout=0):
        super(SeqLabLSTM, self).__init__(h_embs, use_vectors=use_vectors, zero_padding=True)
        assert rnn_hdim > 0 and n_labels > 0 and n_layers > 0

        self.hidden_dim = rnn_hdim
        self.n_layers = n_layers
        self.n_labels = n_labels
        self.n_directions = 2 if bidirectional else 1

        # the big boy LSTM that does all the work
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        # output label prediction at each time step
        self.hidden_to_label = nn.Linear(self.hidden_dim * self.n_directions, self.n_labels)

        # don't do hidden initialization until we know the batch size
        self.hidden = None


    def init_hidden(self, mb_size):
        hstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        cstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        return hstate, cstate


    def forward(self, sorted_tok_ids):
        # get the tensor with emb sequences, along with the number of pads in each seq
        emb_seqs, pads = super(SeqLabLSTM, self).forward(sorted_tok_ids)

        # now we gotta do some special packing
        # note: emb_seqs -> (batch_size, max_seq_len, embedding_dim)
        bsz, max_len, emb_dim = emb_seqs.shape
        X = nn.utils.rnn.pack_padded_sequence(emb_seqs, max_len - pads, batch_first=True)

        # run it, always init the hidden states in this part of the run
        self.hidden = self.init_hidden(bsz)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing to get (batch_size * seq_len_for_each, hidden_dim)
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # now predict the labels
        X = X.contiguous()
        X = X.view(-1, X.shape[2]) # dim is max_len * bsz, hdim

        # run through the linear tag prediction
        X = self.hidden_to_label(X) # dim is max_len * bsz, n_labels

        # softmax activations in the feed forward for an easy main method
        Y_hat = F.log_softmax(X, dim=1)
        return Y_hat.view(bsz, max_len, self.n_labels)


    def loss(self, Y_hat, padded_labels):
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches...
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels, a longtensor of labels (with padding)
        Y = padded_labels.view(-1)
        mask = (Y != self.PADDING_LABEL_ID).float() # create a mask by zeroing out padding tokens

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.n_labels)

        # count how many tokens we have
        n_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        # must do Y - 1 for the real label indexes (pushed from the padding label, which is 0)
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y - 1] * mask

        # compute cross entropy loss which ignores all the pads
        ce_loss = -torch.sum(Y_hat) / n_tokens
        return ce_loss


    def get_label_predictions(self, Y_hat, padded_labels):
        # flatten all the labels, a longtensor of labels (with padding)
        Y = padded_labels.view(-1)
        mask = (Y != self.PADDING_LABEL_ID).long()

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.n_labels)

        # pick the values for the label
        _, preds = torch.max(Y_hat, dim=1)

        # zero out the paddings preds and return the label predictions
        # the plus one is for pushing them back to the label indexes
        return (preds + 1) * mask

