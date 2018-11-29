#!/usr/bin/env bash

hbt_glv_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/hbt_glv.py")')

#python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u10-I500-x100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -I 500 -u 0.1

#python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u1-I1000-x100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -I 1000 -u 0.01

python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u100-I500-x100-l1e5 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -I 500 -u 1

python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u10-I1000-x100-l1e5 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -I 1000 -u 0.1

python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u1-I2000-x100-l1e4 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-4 -I 2000 -u 0.01

python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u0.1-I3000-x100-l1e3 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-3 -I 3000 -u 0.001

python $hbt_glv_path -b w5-harmonic-v10k -o hbt-glv-s1-v10k-u0.01-I5000-x100-l1e2 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-2 -I 5000 -u 0.0001

