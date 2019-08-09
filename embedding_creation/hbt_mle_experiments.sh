#!/usr/bin/env bash
python /home/ndg/users/enewel3/hilbert/hilbert/run_mle.py \
    --bigram 5w-dynamic-40k-giga-wiki \
    --out-dir hbt-mle-v40k-I500-l500-t1-giga-wiki-nohyp \
    --solver adam \
    --seed 1 \
    --learning-rate 500 \
    --iters-per-epoch 500 \
    --epochs 200 \
    --sector-factor 3 \
    --shard-factor 2 \
    --num-loaders 9 \
    --loader-policy buffered-parallel \
    --temperature 1

python /home/ndg/users/enewel3/hilbert/hilbert/run_mle.py \
    --bigram 5w-dynamic-40k-giga-wiki \
    --out-dir hbt-mle-v40k-I500-l250-t1-giga-wiki-nohyp \
    --solver adam \
    --seed 1 \
    --learning-rate 250 \
    --iters-per-epoch 500 \
    --sector-factor 3 \
    --shard-factor 2 \
    --num-loaders 9 \
    --loader-policy buffered-parallel \
    --temperature 1

python /home/ndg/users/enewel3/hilbert/hilbert/run_mle.py \
    --bigram 5w-dynamic-40k-giga-wiki \
    --out-dir hbt-mle-v40k-I500-l100-t1-giga-wiki-nohyp \
    --solver adam \
    --seed 1 \
    --learning-rate 250 \
    --iters-per-epoch 500 \
    --sector-factor 3 \
    --shard-factor 2 \
    --num-loaders 9 \
    --loader-policy buffered-parallel \
    --temperature 1

