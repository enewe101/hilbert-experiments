GIGAWORD=$(python -c 'import shared; print(shared.CONSTANTS.GIGAWORD_DIR)')
LOCAL_GIGAWORD=$(python -c 'import shared; print(shared.CONSTANTS.LOCAL_GIGAWORD_DIR)')

for i in $(ls $GIGAWORD); do
    dirname=$(echo $i | cut -d'.' -f1)
    echo -e "\n\n\t--- STARTING $i ---\n\n"
    tar -zxvf $GIGAWORD/$dirname.tgz -C $LOCAL_GIGAWORD $dirname/CoreNLP
    echo -e "\n\n\t--- DONE $i ---\n\n"
done
