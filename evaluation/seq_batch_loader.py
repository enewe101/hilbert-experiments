import torch
from evaluation.hparams import HParams
from evaluation.torch_model import SeqLabLSTM

PAD_LABEL = SeqLabLSTM.PADDING_LABEL_ID

class SequenceLoader(object):

    def __init__(self, X_seqs, y_labels, mb_size, padding_id, seq_labelling=False):

        # needs to be pre-sorted in decreasing order
        assert len(X_seqs[0]) > len(X_seqs[-1])

        # we will be storing the sequences onto the GPU, sorted into MBs
        self.bsz = mb_size
        self.n_samples = len(X_seqs)
        self.seq_labelling = seq_labelling

        # boxesss
        X_batches = [[]]
        y_batches = [[]]
        batch_pads = [[]]
        crt_max_len = len(X_seqs[0])

        # iterate over everything
        for i, sent in enumerate(X_seqs):

            # add the sentences with the padding
            pads = crt_max_len - len(sent)
            padding = [padding_id] * pads
            padded_sent = sent + padding
            X_batches[-1].append(padded_sent)
            batch_pads[-1].append(pads)

            # if sequence labelling then we need to do the same to label seqs
            if self.seq_labelling:
                padding = [PAD_LABEL] * pads
                padded_labels = y_labels[i] + padding
                y_batches[-1].append(padded_labels)

            # batch is filled!
            if len(X_batches[-1]) == mb_size or i == len(X_seqs) - 1:

                # the batch is complete, push it to the GPU!
                # the batch tensors have the shape: bsz X max_len
                X_batches[-1] = torch.LongTensor(X_batches[-1]).to(HParams.DEVICE)

                # the batch_pads stores the number of pads in each sample in the batch
                batch_pads[-1] = torch.LongTensor(batch_pads[-1]).to(HParams.DEVICE)

                # do seq labels, if necessary
                if self.seq_labelling:
                    y_batches[-1] = torch.LongTensor(y_batches[-1]).to(HParams.DEVICE)

                # move on to building the next one!
                if i < len(X_seqs) - 1:
                    X_batches.append([])
                    batch_pads.append([])
                    crt_max_len = len(X_seqs[i + 1])
                    if self.seq_labelling:
                        y_batches.append([])

        # set the attributes of the loader based on how we just pre-organized the batches
        self.batches = X_batches
        self.batch_pads = batch_pads
        if self.seq_labelling:
            self.y_labels = y_batches
        else:
            self.y_labels = torch.LongTensor(y_labels).to(HParams.DEVICE)


    def __len__(self):
        return len(self.batches)


    def get(self):
        assert len(self) == 1
        ys = self.y_labels[0] if self.seq_labelling else self.y_labels
        return self.batches[0], self.batch_pads[0], ys


    def __iter__(self):
        self.crt_batch = -1
        return self


    def __next__(self):
        self.crt_batch += 1
        if self.crt_batch >= len(self):
            raise StopIteration

        if self.seq_labelling:
            ys = self.y_labels[self.crt_batch]
        else:
            sample_slice = slice(self.crt_batch * self.bsz, (self.crt_batch+1) * self.bsz)
            ys = self.y_labels[sample_slice]

        return self.batches[self.crt_batch], self.batch_pads[self.crt_batch], ys
