#!/bin/bash

#python .download_nltk.py
#python .download_gluon.py

## this repo is released with the wemb_intrinsic datasets
mv .zips/wemb_intrinsic_ds.tar.gz .
tar -xzvf wemb_intrinsic_ds.tar.gz
mv wemb_intrinsic_ds.tar.gz .zips/

mv .zips/wemb_extrinsic_ds.tar.gz .
tar -xzvf wemb_extrinsic_ds.tar.gz
mv wemb_extrinsic_ds.tar.gz .zips/

mkdir np
