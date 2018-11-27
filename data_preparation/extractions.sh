#!/usr/bin/env bash

bigram_extraction_executable=$(python -c 'import data_preparation; print(data_preparation.CONSTANTS.DATA_PREPARATION_SRC_DIR)')/bigram_extraction.py

python $bigram_extraction_executable -c gigaword_tokenized_cat.txt -o w5-harmonic-v10k -s harmonic -w 5 -v 10001
