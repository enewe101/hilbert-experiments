#!/usr/bin/env bash

std_glv_path=$(python -c 'import embedding_creation; print(embedding_creation.CONSTANTS.EMBEDDING_CREATION_SRC_DIR + "/std_glv.py")')

#python $std_glv_path -o std-glv-v10k-iter5-x100 -m 17921 -i 5 -w 5 -t 20 -x 100 -d 300
#python $std_glv_path -o std-glv-v10k-iter10-x100 -m 17921 -i 10 -w 5 -t 20 -x 100 -d 300
#python $std_glv_path -o std-glv-v10k-iter20-x100 -m 17921 -i 20 -w 5 -t 20 -x 100 -d 300
#python $std_glv_path -o std-glv-v10k-iter25-x100 -m 17921 -i 25 -w 5 -t 20 -x 100 -d 300

#python $std_glv_path -o std-glv-v10k-iter10-x1-bias -m 17921 -i 10 -w 5 -t 20 -x 1 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
#python $std_glv_path -o std-glv-v10k-iter10-x10-bias -m 17921 -i 10 -w 5 -t 20 -x 10 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
#python $std_glv_path -o std-glv-v10k-iter10-x100-bias -m 17921 -i 10 -w 5 -t 20 -x 100 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0

python $std_glv_path -o std-glv-v10k-iter5-x10-bias -m 17921 -i 5 -w 5 -t 20 -x 10 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
python $std_glv_path -o std-glv-v10k-iter15-x10-bias -m 17921 -i 15 -w 5 -t 20 -x 10 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
python $std_glv_path -o std-glv-v10k-iter20-x10-bias -m 17921 -i 20 -w 5 -t 20 -x 10 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0

python $std_glv_path -o std-glv-v10k-iter5-x100-bias -m 17921 -i 5 -w 5 -t 20 -x 100 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
python $std_glv_path -o std-glv-v10k-iter15-x100-bias -m 17921 -i 15 -w 5 -t 20 -x 100 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
python $std_glv_path -o std-glv-v10k-iter20-x100-bias -m 17921 -i 20 -w 5 -t 20 -x 100 -d 300 --stats-path std-glv-v10k-iter10-x100 --save-mode 0
