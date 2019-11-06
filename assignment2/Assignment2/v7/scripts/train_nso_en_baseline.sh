#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

python baseline/training.py \
    --cuda \
    --src nso \
    --tgt en \
    --model-file nso-en-baseline.pt \
    --n-layers 4 \
    --n-heads 4 \
    --embed-dim 512 \
    --hidden-dim 512 \
    --dropout 0.3 \
    --word-dropout 0.1 \
    --lr 5e-4 \
    --n-epochs 50 \
    --tokens-per-batch 12000 \
    --clip-grad 0.5

python baseline/translate.py \
    --cuda \
    --src nso \
    --tgt en \
    --model-file nso-en-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/nso_en/nsoen_parallel.bpe.dev.nso \
    --output-file nsoen_parallel.dev.out.en

