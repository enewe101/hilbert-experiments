import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from collections import defaultdict
from sklearn.model_selection import train_test_split
from evaluation.hparams import HParams
from evaluation.results import ResultsHolder


MAX_MB_SIZE = 2048

def sort_by_length(x, y, reverse=False):
    sort_key = lambda xy_tuple: len(xy_tuple[0])
    r = reversed if reverse else lambda a: a # functional programming
    return zip(*r(sorted(zip(x, y), key=sort_key)))


def iter_mb(x, y, minibatch_size):
    for i in range((len(x) // minibatch_size) + 1):
        seqs = x[i * minibatch_size: (i + 1) * minibatch_size]
        labels = y[i * minibatch_size: (i + 1) * minibatch_size]
        yield seqs, torch.LongTensor(labels).to(HParams.DEVICE)


def feed_full_ds(neural_model, minibatch_size, x, y):
    correct = 0
    for tok_seqs, labels in iter_mb(x, y, minibatch_size):
        predictions = neural_model(tok_seqs)
        _, label_preds = torch.max(predictions.data, 1)
        correct += (label_preds == labels).sum().item()
    return correct / len(x)


def train_classifier(exp_name, h_embs, classifier_constr, kw_params,
                     lr, n_epochs, mb_size, early_stop,
                     tr_x, tr_y, te_x, te_y,
                     schedule_lr=False, verbose=True):
    """
    Main function to train any classifier object.
    :param exp_name: name of the experiment (e.g., POS-wsj)
    :param h_embs: HilbertEmbeddings object
    :param classifier_constr: constructor that extends EmbeddingModel
    :param kw_params: dictionary of kwargs
    :param lr: learning rate
    :param n_epochs: max number of epochs to train for
    :param mb_size: size of minibatches, -1 for full batch training
    :param early_stop: number of epochs to stop after no improvement is seen
    :param tr_x: training set X from a Hilbert dataset
    :param tr_y: training set y from a Hilbert dataset
    :param te_x: test set X from a Hilbert dataset
    :param te_y: test set y from a Hilbert dataset
    :param schedule_lr: use a plateau-based scheduled learning rate
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
    model = classifier_constr(h_embs, **kw_params).to(HParams.DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr)
    loss_fun = nn.CrossEntropyLoss()

    # learning rate scheduler to maximize the validation set accuracy.
    # default with a dummy scheduler where no change occurs
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=early_stop // 5, mode='max',
        min_lr=0 if schedule_lr else lr, verbose=True,
    )

    # results storage
    results = defaultdict(lambda: [])
    best_val_acc = 0
    best_epoch = 0
    early_stop_count = early_stop # if performance doesn't improve for 10 epochs, end it

    # determine if we are doing complete batch training
    full_batch_train = mb_size == -1
    if full_batch_train:
        mb_size = MAX_MB_SIZE # large minibatches to go fast (but do not exceed the GPU memory)

    if verbose: print('Beginning training...')
    # now iterate over the epochs
    for e in range(n_epochs):
        if verbose: print('\nEpoch {}: (training)'.format(e))

        # training set iteration
        model.train()
        training_loss = 0
        optimizer.zero_grad()

        # iterate over token sequences and the classification labels for each
        for tok_seqs, labels in iter_mb(tr_x, tr_y, mb_size):

            # check if we are doing full batch trianing, if not, zero-out gradient.
            if not full_batch_train:
                optimizer.zero_grad()

            # make the predictions, compute loss and record it
            predictions = model(tok_seqs)
            loss = loss_fun(predictions, labels)
            training_loss += loss.data.item()

            # compute the back gradient
            loss.backward()

            # if we are not doing full batch training, step the optimizer
            if not full_batch_train:
                optimizer.step()

        # if we are doing full batch training, we do one big step at the end.
        if full_batch_train:
            optimizer.step()

        # even out the loss and record it
        training_loss /= len(tr_x) // mb_size
        results['loss'].append(training_loss)

        # now feed forward and get preds for validation set
        if verbose: print('    (evaluating...)')
        with torch.no_grad():
            model.eval()
            # bigger mbsize for test set because we want to go through it as fast as possible
            train_acc = feed_full_ds(model, max(MAX_MB_SIZE, mb_size), tr_x, tr_y)
            val_acc = feed_full_ds(model, max(MAX_MB_SIZE, mb_size), val_x, val_y)
            test_acc = feed_full_ds(model, max(MAX_MB_SIZE, mb_size), te_x, te_y)

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

                #### Update the LR schedule! ####
                scheduler.step(val_acc)

    # return the results!
    results.update({'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'test_acc_at_best_epoch': results['test_acc'][best_epoch]})
    hresults = ResultsHolder(exp_name)
    hresults.add_ds_results('full', results)
    return hresults


