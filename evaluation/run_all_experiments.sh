#!/usr/bin/env bash

if [[ -z $1 ]]; then
    echo "Must pass argument being the path to the embeddings!"
    exit 1
fi

# glove string
if [[ $1 == *"glv"* ]]; then
    echo "Averaging vectors and covectors!"
    avgvw='--avgvw'
else
    avgvw=''
fi

# set the arguments
constants='--base / --rnn_hdim 128 --n_layers 2 --lr 0.001 --mb_size 16 --hdim1 128 --hdim2 128'
logreg='--model_str logreg'
ffnn='--model_str ffnn --dropout 0.5 --schedule_lr'
bilstm='--model_str bilstm --dropout 0.5 --normalize_gradient --schedule_lr'
pos='--dropout 0.5 --normalize_gradient --schedule_lr'

python run_experiments $1