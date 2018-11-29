#!/usr/bin/env bash

remote_embeddings_dir='/home/rldata/hilbert-embeddings/embeddings'
local_embeddings_dir=$(python -c 'import shared; print(shared.CONSTANTS.EMBEDDINGS_DIR)')
plot_trace_executable=$(python -c 'import shared; print(shared.CONSTANTS.EMBEDDING_CREATION_SRC_DIR)')/plot_trace.py


remote_path=$remote_embeddings_dir/$1/trace.txt
local_dir=$local_embeddings_dir/$1
local_path=$local_dir/trace.txt

mkdir -p $local_dir
scp $remote_path $local_path

python $plot_trace_executable $local_path
