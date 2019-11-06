# Prepare BPE for Extra Monolingual data 
for lang in af nso ts
do
	python baseline/subwords.py train \
	--model_prefix assignment2/AdditionalData/${lang}/Corpora/subwords \
	--vocab_size 8000 \
	--model_type bpe \
	--input assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.${lang}
done

# Apply BPE
for lang in af nso ts
do 
	python baseline/subwords.py segment \
	--model assignment2/AdditionalData/${lang}/Corpora/subwords.model \
	< assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.${lang} \
	> assignment2/AdditionalData/${lang}/Corpora/CORP.NCHLT.CLEAN.BPE.${lang}
done
