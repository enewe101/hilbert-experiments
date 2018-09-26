import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from hilbert_device import DEVICE
from train_classifier import sort_by_length
from torch_model import SeqLabLSTM


def pad_labels(labels):
    # pad the labels here, sequence padding is done in special way later
    max_len = max(len(labels[0]), len(labels[-1]))
    pad = SeqLabLSTM.PADDING_LABEL_ID
    return torch.LongTensor([lab_seq + [pad] * (max_len - len(lab_seq))
                             for lab_seq in labels]).to(DEVICE)


def iter_seq_mb(x, y, minibatch_size):
    for i in range((len(x) // minibatch_size) + 1):
        seqs = x[i * minibatch_size: (i + 1) * minibatch_size]
        labels = y[i * minibatch_size: (i + 1) * minibatch_size]
        if len(seqs) and len(labels):
            yield seqs, labels


def feed_full_seq_ds(neural_model, minibatch_size, x, y):
    # iterate over the dataset to compute the accuracy
    correct, total = 0, 0
    for tok_seqs, label_seq in iter_seq_mb(x, y, minibatch_size):

        # get the yhat prediction matrix for each sample in sequence
        yhat_mat = neural_model(tok_seqs)

        # from that, get the model predictions
        gold_labels = pad_labels(label_seq)
        label_preds = neural_model.get_label_predictions(
            yhat_mat, gold_labels
        )

        # undo the padding
        label_preds = label_preds[label_preds != SeqLabLSTM.PADDING_LABEL_ID]

        # must be same length, if everything went correctly
        gold_labels = gold_labels[gold_labels != SeqLabLSTM.PADDING_LABEL_ID]
        assert len(label_preds) == len(gold_labels)

        # count the correct predictions
        correct += (label_preds == gold_labels).sum().item()
        total += len(label_preds)

    return correct / total


def train_seq_labeller(h_embs, constr, kw_params,
                         lr, n_epochs, mb_size, early_stop,
                         tr_x, tr_y, te_x, te_y,
                         eval_train=True,
                         verbose=True):
    """
    Main function to train any classifier object.
    :param h_embs: HilbertEmbeddings object
    :param constr: constructor that extends EmbeddingModel
    :param kw_params: dictionary of kwargs
    :param lr: learning rate
    :param n_epochs: number of epochs to train for
    :param mb_size: size of minibatches
    :param early_stop: number of epochs to stop after no improvement is seen
    :param tr_x: training set X from a Hilbert dataset
    :param tr_y: training set y from a Hilbert dataset
    :param te_x: test set X from a Hilbert dataset
    :param te_y: test set y from a Hilbert dataset
    :param eval_train: feed full training set after each epoch - expensive!
    :param verbose: if true, display everything at every epoch
    :return: results
    """
    if verbose: print('Intializing model...')
    # first make a separate validation set as 10% of training set
    tr_x, val_x, tr_y, val_y = train_test_split(
        tr_x, tr_y, test_size=0.1, random_state=1848,
    )

    # sort the datasets by length of the sentences, very useful
    tr_x, tr_y = sort_by_length(tr_x, tr_y, reverse=True)
    val_x, val_y = sort_by_length(val_x, val_y, reverse=True)
    te_x, te_y = sort_by_length(te_x, te_y, reverse=True)

    # initialize torch things
    model = constr(h_embs, **kw_params).to(DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr)

    # results storage
    results = defaultdict(lambda: [])
    best_val_acc = 0
    best_epoch = 0
    early_stop_count = early_stop # if performance doesn't improve for 10 epochs, end it

    if verbose: print('Beginning training...')
    # now iterate over the epochs
    for e in range(n_epochs):
        if verbose: print('\nEpoch {}: (training)'.format(e))

        # training set iteration
        training_loss, total = 0, 0
        model.train()
        for tok_seqs, labels in iter_seq_mb(tr_x, tr_y, mb_size):
            yhat = model(tok_seqs)
            loss = model.loss(yhat, pad_labels(labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.data.item()
            total += 1

        training_loss /= total
        results['loss'].append(training_loss)

        # now feed forward and get preds for validation set
        if verbose: print('    (evaluating...)')
        with torch.no_grad():
            model.eval()
            # bigger mbsize for test set because we want to go through it as fast as possible
            train_acc = feed_full_seq_ds(model, 512, tr_x, tr_y) if eval_train else np.nan
            val_acc = feed_full_seq_ds(model, 512, val_x, val_y)
            test_acc = feed_full_seq_ds(model, 512, te_x, te_y)

            for acc, string in zip([train_acc, val_acc, test_acc], ['train', 'val', 'test']):
                results['{}_acc'.format(string)].append(acc)

            # check if it is time to end it
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e
                early_stop_count = early_stop
            else:
                early_stop_count -= 1
                if early_stop_count <= 0:
                    break

            # print results
            if verbose:
                for key in sorted(results.keys()):
                    if 'test' in key: continue
                    print('    {:10} - {:4f}'.format(key, results[key][-1]))

    # return the results!
    results.update({'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'test_acc_at_best_epoch': results['test_acc'][best_epoch]})
    return results


