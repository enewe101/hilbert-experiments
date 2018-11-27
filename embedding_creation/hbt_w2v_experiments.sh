#!/usr/bin/env bash

hbt_w2v_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/hbt_w2v.py")')

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e6-v10k-I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 2.45e-6 -I 100
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc7e6-v10k-I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 7.75e-6 -I 100
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc1e5-v10k=I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 1e-5 -I 100
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e5-v10k-I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 2.45e-5 -I 100

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e6-v10k-I400 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 2.45e-6 -I 400
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc7e6-v10k-I400 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 7.75e-6 -I 400
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc1e5-v10k=I400 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 1e-5 -I 400
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e5-v10k-I400 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-7 -t 2.45e-5 -I 400

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc7e5-v10k-I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-8 -t 7.75e-5 -I 100
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e4-v10k-I100 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-8 -t 2.45e-4 -I 100

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e5-v10k-u10-I4000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 2.45e-5 -I 4000 -u 0.1

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e5-v10k-u10-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 2.45e-5 -I 500 -u 0.1

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc7e6-v10k-u10-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 7.75e-6 -I 500 -u 0.1
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e6-v10k-u10-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 2.45e-6 -I 500 -u 0.1

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e5-v10k-u1-I1000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 2.45e-5 -I 1000 -u 0.01
#
#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc7e6-v10k-u1-I1000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 7.75e-6 -I 1000 -u 0.01

#python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-tc2e6-v10k-u1-I1000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-6 -t 2.45e-6 -I 1000 -u 0.01

python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-t1-v10k-u10-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-9 -t 1 -I 500 -u 0.1

python $hbt_w2v_path -b 5w-dynamic-10k/thresh1 -o hbt-w2v-s1-t1-v10k-u1-I1000 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-8 -t 1 -I 1000 -u 0.01
