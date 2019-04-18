import torch
import evaluation.torch_model as tm


def get_optimizer_scheduler(nn_module, opt_str, early_stop):
    # learning rate scheduler to maximize the validation set accuracy.
    # default with a dummy scheduler where no change occurs

    nn_params = [p for p in nn_module.parameters() if p.requires_grad]

    if opt_str == 'adam':
        optimizer = torch.optim.Adam(nn_params, lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=early_stop // 5, mode='max',
            min_lr=1e-7, verbose=True)

    elif opt_str == 'sgd':
        optimizer = torch.optim.SGD(nn_params, lr=1.0, weight_decay=0.99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=early_stop // 5, mode='max',
            min_lr=1e-6, verbose=True)

    elif opt_str == 'sgd-m':
        optimizer = torch.optim.SGD(nn_params, lr=0.1, weight_decay=0.99,
                                    momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=early_stop // 5, mode='max',
            min_lr=1e-6, verbose=True)

    else:
        raise NotImplementedError('Optimizer scheduler \"{}\" not '
                                  'implemented!'.format(opt_str))
    return optimizer, scheduler


def get_classifier_constr_kwargs(hparams, nclasses):
    neural_kwargs = {'n_classes': nclasses,
                     'fine_tune': hparams.fine_tune,
                     'ffnn': hparams.ffnn}

    mstr = hparams.model_str.lower()

    if mstr == 'logreg':
        neural_constructor = tm.LogisticRegression

    elif mstr == 'ffnn':
        neural_constructor = tm.FFNN
        neural_kwargs.update({'hdim1': hparams.hdim1,
                              'hdim2': hparams.hdim2,
                              'dropout': hparams.dropout})

    elif mstr == 'bilstm':
        neural_constructor = tm.BiLSTMClassifier
        neural_kwargs.update({'rnn_hdim': hparams.rnn_hdim,
                              'n_layers': hparams.n_layers,
                              'dropout': hparams.dropout,
                              'max_pool': False})

    elif mstr == 'bilstm-max':
        neural_constructor = tm.BiLSTMClassifier
        neural_kwargs.update({'rnn_hdim': hparams.rnn_hdim,
                              'n_layers': hparams.n_layers,
                              'dropout': hparams.dropout,
                              'max_pool': True})

    elif mstr == 'att-basic':
        neural_constructor = tm.BasicAttention
        neural_kwargs.update({'learn_W': False,
                              'dropout': hparams.dropout,
                              'distr': hparams.distr_str,
                              'usecovecs': not hparams.nocovecs})

    elif mstr == 'att-diag':
        neural_constructor = tm.BasicAttention
        neural_kwargs.update({'learn_W': True,
                              'diagonal_W': True,
                              'dropout': hparams.dropout,
                              'distr': hparams.distr_str,
                              'usecovecs': not hparams.nocovecs})

    elif mstr == 'att-linear':
        neural_constructor = tm.BasicAttention
        neural_kwargs.update({'learn_W': True,
                              'diagonal_W': False,
                              'dropout': hparams.dropout,
                              'distr': hparams.distr_str,
                              'usecovecs': not hparams.nocovecs})

    elif mstr == 'att-neural':
        neural_constructor = tm.NeuralAttention
        neural_kwargs.update({'dropout': hparams.dropout,
                              'distr': hparams.distr_str,
                              'act': hparams.act_str,
                              'usecovecs': not hparams.nocovecs})

    else:
        raise NotImplementedError('Constructor model \"{}\" not '
                                  'implemented!'.format(mstr))

    return neural_constructor, neural_kwargs

