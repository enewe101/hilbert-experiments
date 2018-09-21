#!/usr/bin/env bash
sectors="$@"
if [ "$sectors" = 'all' ]; then
    sectors=$(for i in {0..9} {a..f}; do for j in {0..9} {a..f}; do for k in {0..9} {a..f}; do echo $i$j$k; done; done; done)
fi

OUT_DIR=$(python -c 'import data_preparation as dp; print(dp.CONSTANTS.DATA_DIR)')/gigaword_tokenized_cat
mkdir -p $OUT_DIR
OUT_PATH=$OUT_DIR/gigaword_tokenized_cat.txt
TOKEN_DIR=$(python -c 'import data_preparation as dp; print(dp.CONSTANTS.TOKENIZED_DIR)')

TOKEN_FILES=$(for sector in $sectors; do echo gathering files for $sector >&2; ls $TOKEN_DIR/$sector/*; done)

#echo $TOKEN_FILES | xargs -n1 echo | echo - yo
rm -f $OUT_PATH
for f in $TOKEN_FILES; do (echo $f >&2; cat "${f}"; echo "") >> $OUT_PATH; done



