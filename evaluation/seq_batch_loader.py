import torch
from evaluation.hparams import HParams

PAD_LABEL = -1


class SequenceLoader(object):

    def __init__(self, X_seqs, y_labels, mb_size, padding_id, seq_labelling=False):

        # needs to be pre-sorted in decreasing order
        assert len(X_seqs[0]) > len(X_seqs[-1])

        # we will be storing the sequences onto the GPU, sorted into MBs
        self.bsz = mb_size
        self.n_samples = len(X_seqs)

        X_batches = [[]]
        X_batch_pads = [[]]
        crt_max_len = len(X_seqs[0])

        # iterate over everything
        for i, sent in enumerate(X_seqs):

            # add the sentences with the padding
            pads = crt_max_len - len(sent)
            padding = [padding_id] * pads
            padded_sent = sent + padding
            X_batches[-1].append(padded_sent)
            X_batch_pads[-1].append(pads)

            if len(X_batches[-1]) == mb_size or i == len(X_seqs) - 1:

                # the batch is complete, push it to the GPU!
                # the batch tensors have the shape: bsz X max_len
                X_batches[-1] = torch.LongTensor(X_batches[-1]).to(HParams.DEVICE)

                # the batch_pads stores the number of pads in each sample in the batch
                X_batch_pads[-1] = torch.LongTensor(X_batch_pads[-1]).to(HParams.DEVICE)

                # move on to building the next one!
                if i < len(X_seqs) - 1:
                    X_batches.append([])
                    X_batch_pads.append([])
                    crt_max_len = len(X_seqs[i + 1])

        self.batches = X_batches
        self.batch_pads = X_batch_pads

        # set the y labels
        if not seq_labelling:
            self.y_labels = torch.LongTensor(y_labels).to(HParams.DEVICE)
        else:
            raise NotImplementedError('Not!')


    def __len__(self):
        return len(self.batches)


    def get(self):
        assert len(self) == 1
        return self.batches[0], self.batch_pads[0], self.y_labels


    def __iter__(self):
        self.crt_batch = -1
        return self


    def __next__(self):
        self.crt_batch += 1
        if self.crt_batch >= len(self):
            raise StopIteration

        sample_slice = slice(self.crt_batch * self.bsz, (self.crt_batch+1) * self.bsz)

        return self.batches[self.crt_batch], self.batch_pads[self.crt_batch], \
               self.y_labels[sample_slice]
