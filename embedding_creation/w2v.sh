#!/usr/bin/env bash
W2V_EXECUTABLE_PATH=$(python -c 'import shared; print(shared.CONSTANTS.W2V_EXECUTABLE_PATH)')

corpus=$1
out_dir=$2
d=$3
window_size=$4
k=$5
t_undersample=$6
threads=$7
num_iterations=$8

vectors_path=$out_dir/vectors.txt
mode=0
hierarchical_sample=0
binary=0

mkdir -p $out_dir

time $W2V_EXECUTABLE_PATH -train $corpus -output $vectors_path -cbow $mode -size $d -window $window_size -negative $k -hs $hierarchical_sample -sample $t_undersample -threads $threads -binary $binary -iter $num_iterations

