# Prepare BPE for Extra Monolingual data 
for lang in af nso ts
do
	mkdir assignment2/AdditionalData/${lang}/Corpora/EN/
	mv assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang} assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang}_EN.${lang} 
	python baseline/subwords.py train \
	--model_prefix assignment2/AdditionalData/${lang}/Corpora/EN/subwords \
	--vocab_size 8000 \
	--model_type bpe \
	--input assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.${lang}_EN.en
done

# Apply BPE
for lang in af nso ts
do 
	python baseline/subwords.py segment \
	--model assignment2/AdditionalData/${lang}/Corpora/EN/subwords.model \
	< assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.${lang}_EN.en \
	> assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang}_EN.en
done
