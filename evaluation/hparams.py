import argparse
import torch
from constants import ALL_DS

class HParams(object):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        parser = argparse.ArgumentParser(description='Run experiments on set of embeddings.')
        parser.add_argument('emb_path', type=str,
                            help='path to the embeddings we want to process')
        parser.add_argument('experiment', type=str, choices=ALL_DS,
                            help='specific experiment to run')
        parser.add_argument('--fine_tune', action='store_true', default=False,
                            help='pass to fine tune embeddings from gradient')
        parser.add_argument('--rnn_hdim', type=int, default=128,
                            help='dimensionality of RNN hidden layer')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='number of layers for a neural net that takes this param')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate for a torch model')
        parser.add_argument('--mb_size', type=int, default=64,
                            help='size of the minibatches')
        parser.add_argument('--dropout', type=float, default=0.,
                            help='dropout probability')
        parser.add_argument('--hdim1', type=int, default=256,
                            help='specific dimensionality of a 1st hidden layer (not for RNNs)')
        parser.add_argument('--hdim2', type=int, default=256,
                            help='specific dimensionality of a 2nd hidden layer (not for RNNs)')
        parser.add_argument('--normalize_gradient', action='store_true', default=False,
                            help='make RNN gradient have max norm equal to 1')
        parser.add_argument('--schedule_lr', action='store_true', default=False,
                            help='flag to use a scheduled learning rate')
        parser.add_argument('--model_str', type=str, choices=['ffnn', 'logreg', 'bilstm'],
                            help='model string for when model selection is applicable')
        args = parser.parse_args()

        # set the namespace of the HParams with those in the args.
        for attr_name, value in vars(args).items():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, value)

        print('Using device: {}'.format(self.DEVICE.type))

        # look and check for required ones for each thing


