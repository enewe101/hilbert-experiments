import torch
import torch.nn as nn
import numpy as np
import unittest
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMTests(unittest.TestCase):

    def test_bilstm(self):
        nlayers = 2
        bsz = 2
        seqlen = 20
        hdim = 4
        embdim = 10

        lstm = nn.LSTM(num_layers=nlayers, batch_first=True, hidden_size=hdim,
                       input_size=embdim, bidirectional=True)

        # init the hidden
        hidden_s = torch.zeros(nlayers * 2, bsz, hdim)
        cell_s = torch.zeros(nlayers * 2, bsz, hdim)

        # make fake inputs and feed through the lstm
        inputs = torch.rand(bsz, seqlen, embdim)
        X, (hidden_s, cell_s) = lstm(inputs, (hidden_s, cell_s))

        # the last two layers of the hidden state contain the final forward and backwards
        last_backward = hidden_s[-1, :, :]
        last_forward = hidden_s[-2, :, :]

        # these should correspond to entries in X, which stores all hidden states
        h_start_seq = X[:, 0, :]
        h_end_seq = X[:, -1, :]

        # concatenate and compare
        concat_h = torch.cat((last_forward, last_backward), dim=1)

        # print(concat_h.shape)
        # print(h_start_seq.shape)
        # print(h_end_seq.shape)
        # print(concat_h == h_start_seq)
        # print(concat_h == h_end_seq)

        # assert that we have the things we want from the BiLSTM
        # e.g., first we compare the last backward state to the hidden states from
        # the start of the sequence - the state where the backward LSTM ends
        # then we compare the last forward state to the hidden state from the end
        # of the sequence, which is exactly what we are looking for.
        self.assertTrue(np.alltrue(last_backward == concat_h[:, hdim:]))
        self.assertTrue(np.alltrue(concat_h[:, hdim:] == h_start_seq[:, hdim:]))

        self.assertTrue(np.alltrue(last_forward == concat_h[:, :hdim]))
        self.assertTrue(np.alltrue(concat_h[:, :hdim] == h_end_seq[:, :hdim]))


    def test_padding(self):
        nlayers = 2
        bsz = 3
        seqlen = 20
        hdim = 5
        embdim = 10

        lstm = nn.LSTM(num_layers=nlayers, batch_first=True, hidden_size=hdim,
                       input_size=embdim, bidirectional=True, bias=False)

        # init the hidden
        hidden_s = torch.zeros(nlayers * 2, bsz, hdim)
        cell_s = torch.zeros(nlayers * 2, bsz, hdim)

        # lets say we have the following paddings, number of pads per seq
        pads = torch.LongTensor([0, 5, 10])
        inputs = torch.rand(bsz, seqlen, embdim)
        for b, pad in zip([1, 2], pads[1:]):
            inputs[b, -pad:, :] = 0.
        X = pack_padded_sequence(inputs, seqlen - pads, batch_first=True)

        # now feed through
        X, (hidden_s, cell_s) = lstm(X, (hidden_s, cell_s))
        last_backward = hidden_s[-1]
        last_forward = hidden_s[-2]
        concat_h = torch.cat((last_forward, last_backward), dim=1)

        # comparing to after undoing padding
        X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = X.contiguous()

        # start will always be the same
        h_start_seq = X[:, 0, :]

        # the end states are at the final pads!
        h_end_0 = X[0, -1, :]
        h_end_1 = X[1, -6, :]
        h_end_2 = X[2, -11, :]
        h_end_seq = torch.stack((h_end_0, h_end_1, h_end_2))

        print(concat_h == h_start_seq)
        print(concat_h == h_end_seq)

        self.assertTrue(np.alltrue(last_backward == concat_h[:, hdim:]))
        self.assertTrue(np.alltrue(concat_h[:, hdim:] == h_start_seq[:, hdim:]))

        self.assertTrue(np.alltrue(last_forward == concat_h[:, :hdim]))
        self.assertTrue(np.alltrue(concat_h[:, :hdim] == h_end_seq[:, :hdim]))

        """
        Conclusion of this test:
        We CAN use the hidden_state produced by the LSTM without running
        pad_packed_sequence. This is because the final states outputted after
        using pack_padded_sequence are the proper end states; i.e., they do
        not include the padding and are indexed properly. 
        """


if __name__ == '__main__':
    unittest.main()

