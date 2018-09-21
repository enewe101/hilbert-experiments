for sector_name in "$@"
do
    echo $sector_name
    GIGAWORD=$(python -c 'import data_preparation as dp; print(dp.CONSTANTS.GIGAWORD_DIR)')
    LOCAL_GIGAWORD=$(python -c 'import data_preparation as dp; print(dp.CONSTANTS.LOCAL_GIGAWORD_DIR)')
    tar -zxvf $GIGAWORD/$sector_name.tgz -C $LOCAL_GIGAWORD $sector_name/CoreNLP
done
