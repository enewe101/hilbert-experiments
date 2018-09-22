#!/usr/bin/env bash
sectors="$@"
if [ "$sectors" = 'all' ]; then
    sectors=$(for i in {0..9} {a..f}; do for j in {0..9} {a..f}; do for k in {0..9} {a..f}; do echo $i$j$k; done; done; done)
fi

OUT_DIR=$(python -c 'import shared; print(shared.CONSTANTS.DATA_DIR)')/gigaword_tokenized_cat
mkdir -p $OUT_DIR
OUT_PATH=$OUT_DIR/gigaword_tokenized_cat.txt
TOKEN_DIR=$(python -c 'import shared; print(shared.CONSTANTS.TOKENIZED_DIR)')

#TOKEN_FILES=$(for sector in $sectors; do echo gathering files for $sector >&2; ls $TOKEN_DIR/$sector/*; done)

#echo $TOKEN_FILES | xargs -n1 echo | echo - yo
rm -f $OUT_PATH

for sector in $sectors; do
    echo gathering files for $sector >&2
    files=$(ls $TOKEN_DIR/$sector/*)
    for f in $files; do 
        (echo $f >&2; cat "${f}" | tr -s [:space:] ' '; echo "") >> $OUT_PATH;
    done
done


#for f in $(for sector in $sectors; do echo gathering files for $sector >&2; ls $TOKEN_DIR/$sector/*; done); do (echo $f >&2; cat "${f}" | tr -s [:space:] ' '; echo "") >> $OUT_PATH; done



