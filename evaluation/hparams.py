import argparse
import torch
import os
from constants import ALL_DS, SUP_DS

KIAN_PATH = '/home/ml/kkenyo1/git/hilbert-experiments/evaluation/test_embs'

class HParams(object):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Run experiments on set of embeddings.')
        parser.add_argument(
            'emb_dir', type=str, nargs='+',
            help='directory[ies] of the embeddings we want to process'
        )
        parser.add_argument(
            'experiment', type=str, choices=ALL_DS,
            help='specific experiment to run'
        )
        parser.add_argument(
            '--base', type=str,
            default=KIAN_PATH,
            help='base directory where the embeddings we will test are stored'
        )
        parser.add_argument(
            '--repeat', type=int, default=1,
            help=(
                "repeat for specified number of times with different seeds "
                "each time (but seeds will always be +1917 from the --seed "
                "parameter)"
            )
        )
        parser.add_argument(
            '--avgvw', action='store_true',
            help='flag to average vecs and covecs (like GloVe does!)'
        )
        parser.add_argument(
            '--seed', type=int, default=1, help='set the random seed')
        parser.add_argument(
            '--epochs', type=int, default=250, help='maximum number of epochs')

        parser.add_argument(
            '--fine_tune', action='store_true', default=False,
            help='pass to fine tune embeddings from gradient'
        )
        parser.add_argument(
            '--rnn_hdim', type=int, default=128,
            help='dimensionality of RNN hidden layer'
        )
        parser.add_argument(
            '--n_layers', type=int, default=2,
            help='number of layers for an RNN'
        )
        parser.add_argument(
            '--lr', type=float, default=0.001,
            help='learning rate for a torch model'
        )
        parser.add_argument(
            '--mb_size', type=int, default=64, help='size of the minibatches')
        parser.add_argument(
            '--dropout', type=float, default=0., help='dropout probability')
        parser.add_argument(
            '--hdim1', type=int, default=256,
            help='specific dimensionality of a 1st hidden layer (not for RNNs)'
        )
        parser.add_argument(
            '--hdim2', type=int, default=256,
            help='specific dimensionality of a 2nd hidden layer (not for RNNs)'
        )
        parser.add_argument(
            '--normalize_gradient', action='store_true', default=False,
            help='make RNN gradient have max norm equal to 1'
        )
        parser.add_argument(
            '--schedule_lr', action='store_true', default=False,
            help='flag to use a scheduled learning rate'
        )
        parser.add_argument(
            '--model_str', type=str, choices=['ffnn', 'logreg', 'bilstm',
                                              'att-basic', 'att-linear', 'att-neural'],
            help='model string for when model selection is applicable'
        )
        # hparams for attention models
        parser.add_argument(
            '--distr_str', type=str, choices=['sigmoid', 'softmax'],
            help='distribution string for attention models'
        )
        parser.add_argument(
            '--act_str', type=str, choices=['sigmoid', 'relu', 'tanh'],
            help='activation function string for model \'att-neural\''
        )
        parser.add_argument(
            '--att_ffnn', action='store_true', default=False,
            help='use a FFNN on top of an attention model, uses 300-d hidden layer'
        )
        parser.add_argument(
            '--normalize', action='store_true', default=False,
            help='normalize the vectors and covectors to unit norm? (rows)'
        )
        parser.add_argument(
            '--standardize', action='store_true', default=False,
            help='normalize the embedding components to be unit norm? (columns)'
        )
        # generics
        parser.add_argument(
            '--dry', action='store_true', 
            help='pass this flag to not run anything'
        )
        parser.add_argument(
            '--data-path', type=str, default='np/all_data.npz',
            help="Location in which to find the evaluation datasets."
        )

        args = parser.parse_args()

        # set the namespace of the HParams with those in the args.
        self.all_params = {}
        for attr_name, value in vars(args).items():
            if not attr_name.startswith('_'):
                setattr(self, attr_name, value)
                self.all_params[attr_name] = value

        print('Using device: {}'.format(self.DEVICE.type))
        if args.dry:
            print('Would be doing experiment {} with each of:'.format(
                self.experiment
            ))
            for p in self.iter_emb_paths():
                print('\t{}'.format(p))
            print('Dry run, exiting...')
            exit(0)

        if self.repeat > 1:
            assert self.experiment in SUP_DS


    def iter_emb_paths(self):
        for emb_dir in self.emb_dir:
            yield os.path.join(self.base, emb_dir)
        return


    def get_params_str(self):
        s = 'Run parameters:\n'
        for key, value in self.all_params.items():
            s += '\t{}: {}\n'.format(key, value)
        return s

