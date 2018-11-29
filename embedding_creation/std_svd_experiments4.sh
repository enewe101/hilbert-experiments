#!/usr/bin/env bash

svd_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/std_svd.py")')

in_screen () { screen -S $1 -dm bash -c "$2; exec sh"; }

##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e6-v10k -t 2.45e-6 -k 15
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e6-v10k -t 7.75e-6 -k 15
##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k -t 1e-5 -k 15
##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k -t 2.45e-5 -k 15
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e5-v10k -t 7.75e-5 -k 15
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e4-v10k -t 2.45e-4 -k 15
##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k -t 1 -k 15
#

#
##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e6-v10k-clipafter -t 2.45e-6 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e6-v10k-clipafter -t 7.75e-6 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k-clipafter -t 1e-5 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k-clipafter -t 2.45e-5 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e5-v10k-clipafter -t 7.75e-5 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e4-v10k-clipafter -t 2.45e-4 -k 15 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k-clipafter -t 1 -k 15 -c
#


##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e6-v10k-clipafter-k10 -t 2.45e-6 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e6-v10k-clipafter-k10 -t 7.75e-6 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k-clipafter-k10 -t 1e-5 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k-clipafter-k10 -t 2.45e-5 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e5-v10k-clipafter-k10 -t 7.75e-5 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e4-v10k-clipafter-k10 -t 2.45e-4 -k 10 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k-clipafter-k10 -t 1 -k 10 -c


#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e6-v10k-clipafter-k5 -t 2.45e-6 -k 5 -c
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e6-v10k-clipafter-k5 -t 7.75e-6 -k 5 -c
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k-clipafter-k5 -t 1e-5 -k 5
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k-clipafter-k5 -t 2.45e-5 -k 5 -c
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e5-v10k-clipafter-k5 -t 7.75e-5 -k 5 -c
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e4-v10k-clipafter-k5 -t 2.45e-4 -k 5 -c
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k-clipafter-k5 -t 1 -k 10 -c



##python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e6-v10k-clipafter-k1 -t 2.45e-6 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e6-v10k-clipafter-k1 -t 7.75e-6 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k-clipafter-k1 -t 1e-5 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k-clipafter-k1 -t 2.45e-5 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t7e5-v10k-clipafter-k1 -t 7.75e-5 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e4-v10k-clipafter-k1 -t 2.45e-4 -k 1 -c
#python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k-clipafter-k1 -t 1 -k 10 -c

