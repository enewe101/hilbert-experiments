#!/bin/bash
mkdir .zips/

#python .download_nltk.py


## this repo is released with the wemb_intrinsic datasets
mv .zips/wemb_intrinsic_ds.tar.gz .
tar -xzvf wemb_intrinsic_ds.tar.gz
mv datasets unsup_datasets
mv wemb_intrinsic_ds.tar.gz .zips/

# download IMBD sentiment analysis dataset
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz --exclude=*.feat
mv aclImdb_v1.tar.gz .zips/


