#!/usr/bin/env bash

hbt_w2v_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/hbt_w2v.py")')

python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-t1-v10k-u1-I2000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-8 -t 1 -I 2000 -u 0.01

python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-t1-v10k-u1-I2000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-8 -t 1 -I 2000 -u 0.001

