#!/usr/bin/env bash

std_w2v_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/std_w2v.py")')

#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t1-v10k-iter5 -t 1 -p 30 -s 1 -l 0.025 -i 5
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s2-t1-v10k-iter5 -t 1 -p 30 -s 2 -l 0.025 -i 5
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t1e5-v10k-iter5 -t 1e-5 -p 30 -s 1 -l 0.025 -i 5
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s2-t1e5-v10k-iter5 -t 1e-5 -p 30 -s 2 -l 0.025 -i 5
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t1e5-v10k-iter2 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 2
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t2e5-v10k-iter5 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 5

#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t2e5-v10k-iter4 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 4
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t2e5-v10k-iter6 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 6
#python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t2e5-v10k-iter8 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 8
python $std_w2v_path -c gigaword-tokenized-cat-10001.txt -o std-w2v-s1-t2e5-v10k-iter10 -t 2.45e-5 -p 30 -s 1 -l 0.025 -i 10
