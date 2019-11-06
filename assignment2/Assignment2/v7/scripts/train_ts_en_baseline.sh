#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

python baseline/training.py \
    --cuda \
    --src ts \
    --tgt en\
    --model-file ts-en-baseline.pt \
    --n-layers 4 \
    --n-heads 4 \
    --embed-dim 512 \
    --hidden-dim 512 \
    --dropout 0.2 \
    --word-dropout 0.2 \
    --lr 1e-3 \
    --n-epochs 30 \
    --tokens-per-batch 12000 \
    --clip-grad 1.0

python baseline/translate.py \
    --cuda \
    --src ts \
    --tgt en \
    --model-file ts-en-baseline.pt \
    --search "beam_search" \
    --beam-size 5 \
    --input-file assignment2/data/ts_en/tsen_parallel.bpe.dev.ts \
    --output-file tsen_parallel.dev.out.en
