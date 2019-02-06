import torch
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from evaluation.train_classifier import sort_by_length
from evaluation.torch_model import SeqLabLSTM
from evaluation.results import ResultsHolder
from evaluation.hparams import HParams
from evaluation.seq_batch_loader import SequenceLoader


MAX_MB_SIZE = 1024


def feed_full_seq_ds(neural_model, seqloader, sst_labels=None):
    doing_sst = sst_labels is not None and False

    # iterate over the dataset to compute the accuracy
    all_gold_labels = []
    all_label_preds = []
    correct, total = 0, 0

    for tok_seqs, pads, label_seq in seqloader:

        # get the yhat prediction matrix for each sample in sequence
        yhat_mat = neural_model(tok_seqs, pads)

        # from that, get the model predictions
        label_preds = neural_model.get_label_predictions(yhat_mat, label_seq)

        # undo the padding
        label_preds = label_preds[label_preds != SeqLabLSTM.PADDING_LABEL_ID]

        # must be same length, if everything went correctly
        gold_labels = label_seq[label_seq != SeqLabLSTM.PADDING_LABEL_ID]
        assert len(label_preds) == len(gold_labels)
        
        # to deal with SST-Labels, where we need to do micro F1 instead to ignore
        # the TP 0 labels, which pollute the accuracy computation
        if doing_sst:
            all_gold_labels += list(gold_labels.cpu().numpy())
            all_label_preds += list(label_preds.cpu().numpy())
        else:
            # count the correct predictions, basic accuracy computation
            correct += (label_preds == gold_labels).sum().item()
            total += len(label_preds)

    if doing_sst:
        return f1_score(all_gold_labels, all_label_preds, labels=sst_labels, average='micro')
    else:
        return correct / total


def train_seq_labeller(exp_name, h_embs, constr, kw_params,
                       lr, n_epochs, mb_size, early_stop,
                       tr_x, tr_y, te_x, te_y,
                       schedule_lr=False,
                       normalize_gradient=False,
                       sst_labels=None,
                       verbose=True):
    """
    Main function to train any classifier object.
    :param exp_name: name of the experiment (e.g., POS-wsj)
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
    :param schedule_lr: use a plateau-based scheduled learning rate
    :param normalize_gradient: perform gradient normalization, making max norm be 1
    :param sst_labels: the relevant labels for supersense tagging (if we're doing it)
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
    model = constr(h_embs, **kw_params).to(HParams.DEVICE)

    # initialize data loaders
    tr_loader = SequenceLoader(tr_x, tr_y, mb_size, model.padding_id, seq_labelling=True)
    val_loader = SequenceLoader(val_x, val_y, MAX_MB_SIZE, model.padding_id, seq_labelling=True)
    te_loader = SequenceLoader(te_x, te_y, MAX_MB_SIZE, model.padding_id, seq_labelling=True)

    # learning rate scheduler to maximize the validation set accuracy.
    # default with a dummy scheduler where no change occurs
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=early_stop // 5, mode='max',
        min_lr=0 if schedule_lr else lr, verbose=verbose
    )
    model_params = [p for p in model.parameters() if p.requires_grad]

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
        for tok_seqs, pads, label_seq in tr_loader:
            optimizer.zero_grad()
            yhat = model(tok_seqs, pads)
            loss = model.loss(yhat, label_seq)
            loss.backward()

            # normalize gradient, if desired
            if normalize_gradient:
                gnorm = clip_grad_norm_(model_params, max_norm=1, norm_type=2)
                results['gnorm'].append(gnorm)

            # do the backprop update
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
            train_acc = feed_full_seq_ds(model, tr_loader, sst_labels)
            val_acc = feed_full_seq_ds(model, val_loader, sst_labels)
            test_acc = feed_full_seq_ds(model, te_loader, sst_labels)

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

            #### Update the LR schedule! ####
            scheduler.step(val_acc)

            # print results
            if verbose:
                for key in sorted(results.keys()):
                    if 'test' in key: continue
                    print('    {:10} - {:4f}'.format(key, results[key][-1]))

    # return the results!
    results.update({'best_epoch': best_epoch,
                    'best_validation_accuracy': best_val_acc,
                    'test_accuracy_at_best_epoch': results['test_acc'][best_epoch]})

    hilb_res = ResultsHolder(exp_name)
    hilb_res.add_ds_results('full', results)
    return hilb_res


