#!/usr/bin/env bash

svd_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/svd.py")')

python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1e5-v10k -t 1e-5 -k 15
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t2e5-v10k -t 2.45e-5 -k 15
python $svd_path -b 5w-dynamic-10k/thresh1 -o std-svd-t1-v10k -t 1 -k 15


