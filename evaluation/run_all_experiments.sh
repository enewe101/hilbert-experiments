#!/usr/bin/env bash

if [[ -z $1 || -z $2 ]]; then
    echo "Must pass argument being the path to the embeddings!"
    echo "Must also pass argument indicating the experiment to run!"
    exit 1
fi

# check if we need a model string as well
if [[ $2 == "sentiment" || $2 == "news" ]]; then
    if [[ -z $3 ]]; then
        echo "For sentiment or news you must also pass model string!"
        exit 1
    elif [[ $3 != "bilstm" && $3 != "ffnn"  && $3 != "logreg" ]]; then
        echo "ERROR: you did not pass a correct model string!"
        echo "You passed: $3. Must be one of: bilstm, ffnn, logreg."
        exit 1
    fi
fi

# glove string
if [[ $1 == *"glv"* ]]; then
    echo "Averaging vectors and covectors!"
    avgvw="--avgvw"
else
    avgvw=""
fi

# set the arguments
mlargs="--lr 0.001 --mb_size 16 --epochs 250"
pos="--dropout 0.5 --normalize_gradient --schedule_lr --rnn_hdim 128 --n_layers 2"

if [[ $3 == "logreg" ]]; then
    modelstr="--model_str logreg"
elif [[ $3 == "ffnn" ]]; then
    modelstr="--model_str ffnn --dropout 0.5 --schedule_lr --hdim1 128 --hdim2 128"
elif [[ $3 == "bilstm" ]]; then
    modelstr="--model_str bilstm $pos"
fi

# analogy experiments first
start="python run_experiments.py $1 $2 --base /"

if [[ $2 == "analogy" ]]; then
    cmd="$start"

elif [[ $2 == "brown-pos" || $2 == "wsj-pos" ]]; then
    cmd="$start $mlargs $pos --repeat 10"

elif [[ $2 == "news" || $2 == "sentiment" ]]; then
    cmd="$start $mlargs $modelstr --repeat 10"

else
    echo "$2: INVALID EXPERIMENT! No such experiment exists."
    exit 1
fi

#echo "We will be running the following command:"
echo $cmd
#$cmd
