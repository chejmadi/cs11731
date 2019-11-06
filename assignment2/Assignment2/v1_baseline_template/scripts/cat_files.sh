cd assignment2
for lang in af ts nso
do
	cat data/en_${lang}/en${lang}_parallel.bpe.train.${lang} AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang}_EN.${lang} > data/en_${lang}/en${lang}_parallelplus.bpe.train.${lang} 
	cat data/en_${lang}/en${lang}_parallel.bpe.train.en AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang}_EN.en > data/en_${lang}/en${lang}_parallelplus.bpe.train.en 
done 
