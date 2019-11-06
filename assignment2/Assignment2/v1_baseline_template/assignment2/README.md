For assignment 2, you are going to explore approaches to improve NMT systems that translate English into a low-resource language. Specifically, we focus on three languages from Africa: Xitsonga (ts), Northern Sotho (nso), Afrikaans (af). Low-resource translation is an active research area in NMT, since neural models tend to under-perform statistical method with limited data[1]. It is particularly interesting to study NMT systems that translate into a low-resource language other than English, as it allows us to improve systems that can better serve the minority languages and the people who speak them.   

## Dataset

We provide the train/dev/test sets for the three languages under the folder `data/`. These are untokenized raw data. For this assignment, you can use your own pre-processing strategies, such as sentencepiece or data cleaning strategies, to improve the NMT system. Our data is from the ukuxhumana project (https://github.com/LauraMartinus/ukuxhumana).

## Output

We only provide the source side (in English) of the test sets. You are expected to use the dev sets for model evaluation, decode the test set using the best model, and submit the decoded outputs (You should document the dev set BLEU scores, and its corresponding test decoding files. We will evaluate the test set performance). The format of the test set decoding should be a standard text file, with one translation sentence per line and in the same order as the source sentences.

## Baseline

We will release a baseline number based on a Transformer model, using only the training data provided. If your code from assignment 1 performs quite well, you are welcomed to implement this assignment based on your own code. If not, you can use our Transformer model, which will be released soon. Either way, the focus of this assignment is to come up with specific strategies to improve the NMT performance under low-resource settings.   
 
## Methods

For this project, we expect you to explore interesting ideas that can improve low-resource NMT. Some potential ideas include: 1) use data from related langauges that are similar to the low-resource language[2]. You can use other publically available datasets, but make sure to cite them; 2) some data cleaning methods, or intelligent data augmentation strategies such as back-translation etc.; 3) modifications to the model architecture, or adding other information, such as syntax, to help model generalize better.  

It is good to try out some methods proposed in recent research papers. An easy way to get started with literature reading is to search for NMT papers in the recent proceedings of ACL, EMNLP, and NAACL. We encourage you to be creative and come up with your own ideas! Even if it didn't work out as well, it is valuable to document the motivation behind your idea, the results, and analysis of why you think it doesn't perform as well.

[1] Six Challenges for Neural Machine Translation. Philipp Koehn and Rebecca Knowles. In ACL 2017.
[2] Rapid Adaptation of Neural Machine Translation to New Languages. Graham Neubig and Junjie Hu. In EMNLP 2018.
