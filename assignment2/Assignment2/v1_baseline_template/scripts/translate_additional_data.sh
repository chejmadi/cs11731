#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0

for lang in af ts nso
do
	echo "Translating ${lang} to English"
	python baseline/translate.py --cuda --src ${lang} --tgt en --model-file ${lang}-en-backtranslate.pt --search "beam_search" --beam-size 5 --input-file assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang} --output-file assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.${lang}_EN.en
done
