import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluation.torch_model_base as tmb
from evaluation.hparams import HParams


# Most simple type of embedding pooler, allowing for mean-max pooling.
class BasicPooling(tmb.EmbeddingModel):

    def __init__(self, h_embs, n_classes,
                 pooling='max',
                 dropout=0.,
                 ffnn=True,
                 **kwargs):

        super(BasicPooling, self).__init__(h_embs, store_covecs=False,
                                           zero_padding=True, **kwargs)
        self.covecs = False
        self.do_mean = pooling == 'mean'
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = tmb.MLPClassifier(self.emb_dim, n_classes,
                                            dropout=dropout, nonlinear=ffnn)


    def forward(self, token_seqs, pads=None):
        assert (pads is not None)
        embs, _ = super(BasicPooling, self).forward(token_seqs) # shape is B x L x d
        B, L, d = embs.shape
        embs = self.dropout(embs)

        if self.do_mean:
            embs = torch.sum(embs, dim=1)
            embs = (embs.t() / (L - pads.float())).t()

        else: # do max
            embs = torch.max(embs, dim=1)[0]

        y = self.classifier(embs)
        return F.log_softmax(y, dim=1).squeeze()



# Sequences transformer network with attention, for classification.
class BasicAttention(tmb.EmbeddingModel):

    def __init__(self, h_embs, n_classes,
                 learn_W=False,
                 diagonal_W=False,
                 dropout=0,
                 distr='softmax',
                 ffnn=True,
                 usecovecs=True,
                 **kwargs):

        super(BasicAttention, self).__init__(h_embs,
              zero_padding=True, store_covecs=True, **kwargs)

        # most important here is how we parameterize the internal W
        # matrix that augments vector-covector dot products.
        # So, W is either a dxd matrix or d-dimensional vector.
        if learn_W:
            self.W = tmb.rvector(self.emb_dim) if diagonal_W else \
                     tmb.rmatrix(self.emb_dim)
            self.W = nn.Parameter(self.W, requires_grad=True)
        else:
            self.W = torch.ones(self.emb_dim).to(HParams.DEVICE)

        self.usecovecs = usecovecs # for testing without them
        self.n_classes = n_classes
        self.vasawani = torch.sqrt(tmb.torch_scalar(self.emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.distr = tmb.get_distr_fun(distr)
        self.classifier = tmb.MLPClassifier(2 * self.emb_dim, n_classes,
                                            dropout=dropout, nonlinear=ffnn)


    def forward(self, token_seqs, pads=None):
        vec_seqs, covec_seqs = super(BasicAttention, self).forward(
            token_seqs, get_covecs=True
        )
        assert (vec_seqs.shape == covec_seqs.shape)
        assert (pads is not None)

        # note: seqs -> (batch_size, max_seq_len, embedding_dim)
        bsz, max_len, emb_dim = vec_seqs.shape

        # here we are doing dropout, but note we will be incidentally
        # dropping out padding components too, but that doesn't matter.
        Ks = self.dropout(vec_seqs) # B x L x d
        Qs = self.dropout(covec_seqs) if self.usecovecs else \
             self.dropout(vec_seqs) # B x L x d

        # get energy matrices with batch-wise multiplications,
        # after doing the W map on to K and then transposing the Qs
        # i.e., .transpose(1,2) -> B x d x L
        # note that the w-map can be parameterized as a diagonal matrix,
        # and thus just an element wise vector multiplication.
        mapped_Ks = Ks @ self.W if len(self.W.shape) == 2 else Ks * self.W
        Es = torch.bmm(mapped_Ks, Qs.transpose(1, 2)) / self.vasawani

        # get the mask tensor and apply it to the energy matrix!
        Es = tmb.apply_energy_mask(Es, bsz, max_len, pads)

        # now get attention matrices from across the batch
        Ak = self.distr(torch.max(Es, dim=1)[0]).reshape(bsz, 1, max_len)
        Aq = self.distr(torch.max(Es, dim=2)[0]).reshape(bsz, 1, max_len)

        # note that Ak and Aq are both B x L
        # or, each holds B attention vectors (of length L)
        vK = (Ak @ mapped_Ks).squeeze() # B x 1 x L times a B x L x d
        vQ = (Aq @ Qs).squeeze()

        # basically done, just conccat and then predict!
        X = torch.cat((vK, vQ), dim=1)
        y = self.classifier(X)
        return F.log_softmax(y, dim=1).squeeze()



# Sequences transformer network with attention, for classification
class NeuralAttention(tmb.EmbeddingModel):

    def __init__(self, h_embs, n_classes,
                 dropout=0,
                 act='sigmoid',
                 distr='softmax',
                 ffnn=True,
                 usecovecs=True,
                 **kwargs):

        super(NeuralAttention, self).__init__(
            h_embs, zero_padding=True, store_covecs=True, **kwargs
        )
        self.usecovecs = usecovecs

        # matrices for neural transformations
        self.Wk = nn.Parameter(tmb.rmatrix(self.emb_dim))
        self.Wq = nn.Parameter(tmb.rmatrix(self.emb_dim))
        self.Wvk = nn.Parameter(tmb.rmatrix(self.emb_dim))
        self.Wvq = nn.Parameter(tmb.rmatrix(self.emb_dim))

        # other aspects
        self.n_classes = n_classes
        self.vasawani = torch.sqrt(tmb.torch_scalar(self.emb_dim))
        self.distr = tmb.get_distr_fun(distr)
        self.act = tmb.get_act_fun(act)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = tmb.MLPClassifier(2 * self.emb_dim, n_classes,
                                            dropout=dropout, nonlinear=ffnn)


    def forward(self, token_seqs, pads=None):
        vec_seqs, covec_seqs = super(NeuralAttention, self).forward(
            token_seqs, get_covecs=True
        )
        assert (vec_seqs.shape == covec_seqs.shape)
        assert (pads is not None)

        # note: seqs -> (batch_size, max_seq_len, embedding_dim)
        bsz, max_len, emb_dim = vec_seqs.shape

        # do the big batch multiplication all at once
        Ks = self.dropout(vec_seqs)
        eKs = self.act(Ks @ self.Wk) # energized neural mapping
        Qs = self.dropout(covec_seqs) if self.usecovecs else \
             self.dropout(vec_seqs)
        eQs = self.act(Qs @ self.Wq) # energized neural mapping

        # now get the energy matrices
        Es = torch.bmm(eKs, eQs.transpose(1,2)) / self.vasawani

        # get the mask tensor and apply it to the energy matrix!
        Es = tmb.apply_energy_mask(Es, bsz, max_len, pads)

        # now get attention matrices from across the batch
        Ak = self.distr(torch.max(Es, dim=1)[0]).reshape(bsz, 1, max_len)
        Aq = self.distr(torch.max(Es, dim=2)[0]).reshape(bsz, 1, max_len)

        # note that Ak and Aq are both B x L. Another interpretation:
        #  each holds B attention vectors (of length L)
        # We will now multiply attention on the attended neural mappings of K & Q
        # i.e., we are defining the Values with different neural transformations
        # and are now attending upon them
        vK = (Ak @ self.act(Ks @ self.Wvk)).squeeze() # B x 1 x L times a B x L x d
        vQ = (Aq @ self.act(Qs @ self.Wvq)).squeeze()

        # basically done, just conccat and then predict!
        X = torch.cat((vK, vQ), dim=1)
        y = self.classifier(X)
        return F.log_softmax(y, dim=1).squeeze()



# Basic BiLSTM for classification of word sequences.
class BiLSTMClassifier(tmb.EmbeddingModel):

    def __init__(self, h_embs, n_classes, rnn_hdim,
                 n_layers=1, dropout=0, max_pool=False,
                 ffnn=True, **kwargs):

        super(BiLSTMClassifier, self).__init__(h_embs, zero_padding=True, **kwargs)
        assert rnn_hdim > 0 and n_classes > 0 and n_layers > 0

        self.hidden_dim = rnn_hdim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.max_pool = max_pool

        # the big boy LSTM that does all the work
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # output label prediction at each time step
        self.classifier = tmb.MLPClassifier(2 * self.hidden_dim, n_classes,
                                            dropout=dropout, nonlinear=ffnn)

        # don't do hidden initialization until we know the batch size
        self.hidden = None


    def init_hidden(self, mb_size):
        hstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        cstate = torch.zeros(self.n_layers * 2, mb_size, self.hidden_dim).to(HParams.DEVICE)
        return hstate, cstate


    def forward(self, token_seqs, pads=None):
        # get the tensor with emb sequences, along with the number of pads in each seq
        emb_seqs, _ = super(BiLSTMClassifier, self).forward(token_seqs)

        # now we gotta do some special packing
        # note: emb_seqs -> (batch_size, max_seq_len, embedding_dim)
        bsz, max_len, emb_dim = emb_seqs.shape
        self.hidden = self.init_hidden(bsz)
        X = nn.utils.rnn.pack_padded_sequence(emb_seqs, max_len - pads, batch_first=True)

        # feed throught the bilstm
        X, (hidden_state, cell_state) = self.lstm(X, self.hidden)

        # The max-pooling is from Conneau et al.'s 2017 EMNLP paper on sentence embedding.
        if self.max_pool:
            X = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)[0]

            # shape of X is B x L x 2h, so need to max-pool along dim=1, L
            X = torch.max(X, dim=1)[0]
            assert (X.shape == (bsz, self.hidden_dim * 2))

        # Otherwise, do first-last concatenation, as done in our ICML paper.
        # This facilities if we are using multiple layered-LSTMs, so that's why
        # we grab from the hidden state using negative indices.
        else:
            last_backward = hidden_state[-1, :, :]
            last_forward = hidden_state[-2, :, :]
            X = torch.cat((last_forward, last_backward), dim=1)

        # run through the classifier
        y = self.classifier(X)
        return F.log_softmax(y, dim=1).squeeze()




####### Sequenece Labelling Models! Just one, BiLSTM #########

# LSTM for sequence labelling (POS-tagging & SS-tagging)
class SeqLabLSTM(tmb.EmbeddingModel):

    # universal constant, do not change!
    PADDING_LABEL_ID = 0


    # extends the EmbeddingModel class which uses our pretrained embeddings.
    def __init__(self, h_embs, n_labels, rnn_hdim, n_layers=1,
                 bidirectional=True, dropout=0, **kwargs):
        super(SeqLabLSTM, self).__init__(h_embs, zero_padding=True, **kwargs)
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


    def forward(self, sorted_tok_ids, pads):
        # get the tensor with emb sequences, along with the number of pads in each seq
        emb_seqs, _ = super(SeqLabLSTM, self).forward(sorted_tok_ids)

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

