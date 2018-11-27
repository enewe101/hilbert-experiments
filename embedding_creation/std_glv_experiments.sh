#!/usr/bin/env bash

std_glv_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/std_glv.py")')

python $std_glv_path -o std-glv-v10k-iter5-x100 -m 17921 -i 5 -w 5 -t 20 -x 100 -d 300
python $std_glv_path -o std-glv-v10k-iter10-x100 -m 17921 -i 10 -w 5 -t 20 -x 100 -d 300
python $std_glv_path -o std-glv-v10k-iter20-x100 -m 17921 -i 20 -w 5 -t 20 -x 100 -d 300
python $std_glv_path -o std-glv-v10k-iter25-x100 -m 17921 -i 25 -w 5 -t 20 -x 100 -d 300
