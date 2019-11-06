#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

python baseline/training.py \
    --cuda \
    --src af \
    --tgt en \
    --model-file af-en-baseline.pt \
    --n-layers 4 \
    --n-heads 4 \
    --embed-dim 512 \
    --hidden-dim 512 \
    --dropout 0.3 \
    --word-dropout 0.1 \
    --lr 1e-3 \
    --n-epochs 50 \
    --tokens-per-batch 4000 \
    --clip-grad 1.0

python baseline/translate.py \
    --cuda \
    --src af \
    --tgt en \
    --model-file af-en-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/af_en/afen_parallel.bpe.dev.af \
    --output-file afen_parallel.dev.out.en

python baseline/translate.py \
    --cuda \
    --src af \
    --tgt en \
    --model-file af-en-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/af_en/afen_parallel.bpe.test.af \
    --output-file afen_parallel.test.out.en
