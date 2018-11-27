
hbt_svd_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/hbt_svd.py")')


# TRY VARIOUS UNDERSAMPLING VALUES
python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e6-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-6 -I 500

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc7e6-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 7.75e-6 -I 500

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc1e5-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 1e-5 -I 500

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e5-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-5 -I 500

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc7e5-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 7.75e-5 -I 500

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e4-v10k-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-4 -I 500


## TRY VARIOUS MINIBATCH FRACTIONS
python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e5-v10k-u10-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-5 -I 500 -u 0.1

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e5-v10k-u1-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-5 -I 500 -u 0.01

python $hbt_svd_path -b 5w-dynamic-10k/thresh1 -o hbt-svd-s1-tc2e5-v10k-u0.1-I500 -i std-w2v-s1-t1-v10k-iter5/vectors-init -S 1 -s sgd -l 1e-5 -t 2.45e-5 -I 500 -u 0.001
