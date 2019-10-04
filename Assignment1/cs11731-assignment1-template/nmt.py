# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --attn-style=<int>                      whether to choose dot-product or MLP attention [default: 0]
    --pretrained-model=<str>                path to pretrained model, if it exists [default: None]
    --weight-decay=<float>                  what weight decay to use [default: 0.00001]

"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import copy
# print (lr, float(args['--lr-decay']))

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])



class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, attn_style, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.ip_embeds = nn.Embedding(len(self.vocab.src), self.embed_size)
        self.lstm_enc = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)
        self.lstm_dec_ipfeeding = nn.LSTM((self.hidden_size+self.embed_size), self.hidden_size)
        self.lstm_dec = nn.LSTM(self.embed_size, self.hidden_size)
        self.dec_lin = nn.Linear(2*self.hidden_size, len(self.vocab.tgt))
        self.op_embeds = nn.Embedding(len(self.vocab.tgt), self.embed_size)
        self.sm = nn.Softmax(dim = 1) # We'll check this later. Dim = 1 for attention.
        self.sm_decoding = nn.Softmax(dim=2)
        self.attn_style = attn_style # DotProd(1), MLP(0), etc.
        self.tanh = nn.Tanh() # For Attention MLP
        self.MLPLin1 = nn.Linear(2*self.hidden_size, self.hidden_size) # Attention MLP layer 1
        self.MLPLin2 = nn.Linear(self.hidden_size, 1) # Attention MLP Layer 2
        self.drop = nn.Dropout(p = self.dropout_rate) 
        # initialize neural network layers...

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        # src_embeddings = self.embed(src_sents)
        src_encodings, decoder_init_state, mask = self.encode(src_sents)
        scores, target = self.decode(src_encodings, decoder_init_state, tgt_sents, mask)
        return scores, target

    def encode(self, src_sents: List[List[str]]): # -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        try:
            src_tok = torch.tensor([[self.vocab.src.word2id[i] for i in c]for c in src_sents], dtype=torch.long) # batch_size x seq_len_src
        except KeyError:
            src_tok = []
            for c in src_sents:
                for i in c:
                    if i not in self.vocab.src:
                        i = '<unk>'
                    src_tok.append(self.vocab.src.word2id[i])
            src_tok = torch.tensor([src_tok], dtype=torch.long)
                        
        src_tok = torch.t(src_tok) # seq_len_src x batch_size
        mask = torch.ones(src_tok.shape).cuda()
        mask[src_tok == 0] = 0 # Boolean mask created where indices corresponding to <pad> are zeroes.
        src_embeds = self.ip_embeds(src_tok.cuda()) # seq_len_src x batch_size x embed_size
        src_encodings, last_hidden = self.lstm_enc(src_embeds) 
        # src_encodings: seq_len_src x batch_size x (2 * hidden_size), last_hidden: (2 x batch_size x hidden_size, 2 x batch_size x hidden_size)
        seq_len_src = src_encodings.shape[0]
        batch_size = src_encodings.shape[1]
        src_encodings = src_encodings.view(seq_len_src, batch_size, 2, self.hidden_size).sum(dim=2) 
        decoder_init_state = (last_hidden[0].sum(dim = 0).unsqueeze(0), last_hidden[1].sum(dim = 0).unsqueeze(0))
        # src_encodings: seq_len_src x batch_size x hidden_size, decoder_init_state: (1 x batch_size x hidden_size, 1 x batch_size x hidden_size)
        return src_encodings, decoder_init_state, mask 

    def decode(self, src_encodings, decoder_init_state, tgt_sents: List[List[str]], mask):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        # Get embeddings for the target sentence
        tgt_tok = torch.tensor([[self.vocab.tgt.word2id[i] for i in c] for c in tgt_sents], dtype=torch.long)
        tgt_tok = torch.t(tgt_tok) # Get it in PyTorch format : seq_len_tgt x batch_size
        tgt_embeddings = self.op_embeds(tgt_tok.cuda()) # seq_len_tgt x batch_size x embed_size
        tgt_embeddings = tgt_embeddings[:(len(tgt_embeddings)-1)]
        # print (len(tgt_embeddings))

        # Run decoder lstm cell for every word in the target
        hidden = decoder_init_state # Initialise the hidden state and memory cell for the first LSTM Cell of the decoder
        # dec_ip = tgt_embeddings[0] # Initialise the input as the embedding of the SOS token.

        if True:   
            # scores = torch.tensor([[[]]]).cuda()    
            sos = True
            context = torch.zeros([1, tgt_embeddings.shape[1], self.hidden_size]).cuda() # Initialise context vector
            for word in tgt_embeddings:
                word = word.unsqueeze(0) 
                output, hidden = self.lstm_dec_ipfeeding(torch.cat((word, context), dim=2), hidden) # output: 1 x batch_size x hidden_size, hidden = 1 x batch_size x hidden_size
                context, attn_vctr = self.attention(src_encodings, output, mask, self.attn_style) 
                # context: 1 x batch_size x hidden_size, attn_vctr: batch_size x seq_len_src x 1
                attended_output = torch.cat((output, context), dim = 2) # attended_output: 1 x batch_size x (2*hidden_size) 
                dropped_output = self.drop(attended_output)
                new_scores = self.dec_lin(dropped_output) # new_scores = 1 x batch_size x tgt_vocab_size
                if sos:
                    scores = new_scores
                    sos = False
                else:
                    scores = torch.cat((scores, new_scores), dim = 0)



        # count = 0 
        # for word in tgt_embeddings:
        #     hx, cx = self.lstm_dec(word, hidden)
        #     hidden = hx
        #     lin = self.dec_lin(hx)
        #     probs = self.sm(lin) 
        #     scores.append(SomeLossFn(probs[tgt_tok[count]])) # Pseudo code
        #     count += 1
        else:
            output, _ = self.lstm_dec(tgt_embeddings, hidden) # output: seq_len_tgt x batch_size x hidden_size
            context, attn_vctr  = self.attention(src_encodings, output, mask, self.attn_style) # context: seq_len_tgt x batch_size x hidden_size
            attended_output = torch.cat((output, context), dim = 2) # attended_output: seq_len_tgt x batch_size x 2*hidden_size
            dropped_output = self.drop(attended_output)
            scores = self.dec_lin(dropped_output) # scores: seq_len_tgt x batch_size x tgt_vocab_size 
            # scores = np.sum(SomeLossFn(probs[tgt_tok])) # We need to check this
        return scores, tgt_tok

    def beam_search(self, src_sent: List[str], beam_size: int=1, max_decoding_time_step: int=70):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        with torch.no_grad():
            self.eval()
            # Encode
            src_encodings, decoder_init, mask = self.encode([src_sent]) 
            hypotheses = []
            
            # Greedy decoding
            word = torch.tensor([[1]], dtype=torch.long) # SOS - 1 x 1
            context = torch.zeros([1, 1, self.hidden_size]).cuda()
            stopping_criterion = False
            total_score = 0
            sentence = []
            count = 0
            hidden = decoder_init
                
            if beam_size == 1:
                while not stopping_criterion: 
                    word_embed = self.op_embeds(torch.t(word).cuda()) # 1 x 1 x embed_size
                    if True:
                        output, hidden = self.lstm_dec_ipfeeding(torch.cat((word_embed, context), dim=2), hidden) 
                    else:
                        output, hidden = self.lstm_dec(word_embed, hidden)
                    # output: 1 x 1 x hidden_size
                    context, attn_vctr = self.attention(src_encodings, output, mask, self.attn_style) 
                    # context: 1 x 1 x hidden_size, attn_vctr: 1 x seq_len_src x 1
                    attended_output = torch.cat((output, context), dim = 2) # 1 x 1 x (2*hidden_size)
                    scores = self.dec_lin(attended_output) # 1 x 1 x tgt_vocab_size
                    scores = self.sm_decoding(scores) # 1 x 1 x tgt_vocab_size
                    score, word = torch.topk(scores, 1, 2) # Pick the best 
                    word = word.squeeze(0)
                    total_score += np.log(score.item())
                    if word.item() == 3: # unk replacment
                        sentence.append(src_sent[torch.argmax(attn_vctr, dim = 1).item()])
                    else:
                        sentence.append(self.vocab.tgt.id2word[word.item()])
                    count += 1
                    if '</s>' in sentence:
                        sentence.remove('</s>')
                        stopping_criterion = True
                    if count == max_decoding_time_step:
                        stopping_criterion = True 

                hypothesis = Hypothesis(sentence, total_score)
                hypotheses.append(hypothesis)
            else:
                # Proper beam search
                # At the first step, we do normal decoding from the start of sentence token
                #  to get a batch of: k words and scores, 1 x k tokens
                word_embed = self.op_embeds(torch.t(word).cuda()) # 1 x 1 x embed_size
                if True:
                    output, hidden = self.lstm_dec_ipfeeding(torch.cat((word_embed, context), dim=2), hidden) 
                else:
                    output, hidden = self.lstm_dec(word_embed, hidden)
                 # output: 1 x 1 x hidden_size
                context, attn_vctr = self.attention(src_encodings, output, mask, self.attn_style) 
                # context: 1 x 1 x hidden_size, attn_vctr: 1 x seq_len_src x 1
                attended_output = torch.cat((output, context), dim = 2) # 1 x 1 x (2*hidden_size)
                scores = self.dec_lin(attended_output) # 1 x 1 x tgt_vocab_size
                scores = self.sm_decoding(scores) # 1 x 1 x tgt_vocab_size 
                probs, tokens = torch.topk(scores, beam_size, 2) # 1 x 1 x k
                log_probs = torch.from_numpy(np.log(probs.numpy())) # 1 x 1 x k
                context = context.repeat(1, beam_size, 1)
                hidden = (hidden[0].repeat(1, beam_size, 1), hidden[1].repeat(1, beam_size,1))
                sentences = [[self.vocab.tgt.id2word[i.item()]] for i in tokens.squeeze(0).squeeze(0)]
                # We now have our first words in each of k hypotheses
                # Now we can enter the while loop
                stopping_criterion = False 
                hypo = [] # Actual final hypotheses
                final_scores = []
                while not stopping_criterion:
                    word_embed = self.op_embeds(torch.t(tokens).cuda()) # 1 x k x embed_size
                    if True:
                        output, hidden = self.lstm_dec_ipfeeding(torch.cat((word_embed, context), dim=2), hidden) 
                    else:
                        output, hidden = self.lstm_dec(word_embed, hidden)
                    # output: 1 x k x hidden_size
                    context, attn_vctr = self.attention(src_encodings, output, mask, self.attn_style) 
                    # context: 1 x k x hidden_size, attn_vctr: k x seq_len_src x 1
                    attended_output = torch.cat((output, context), dim = 2) # 1 x k x (2*hidden_size)
                    scores = self.dec_lin(attended_output) # 1 x k x tgt_vocab_size
                    scores = self.sm_decoding(scores) # 1 x k x tgt_vocab_size 
                    # Add previous candidate scores
                    total_scores = torch.add(log_probs.permute(0,2,1), torch.from_numpy(np.log(scores))) # 1 x k x vocab_size
                    # Flatten the total_scores and pick top k 
                    log_probs, candidates = torch.topk(total_scores.view(1, 1, -1)), beam_size, 2) # 1 x 1 x k
                    parents = candidates//total_scores.shape[2] # Provenance of new tokens
                    tokens = candidates%total_scores.shape[2] # Actual new word tokens
                    hidden = (hidden[0][0][parents], hidden[1][0][parents]) # Get hidden state to pass on
                    context = context[0][parents] # Get correct context to pass on
                    attn_vctr = attn_vctr[parents] # Keep correct attention vector for unk replacement
                    # Create copies of the above to delete in case EOS appears in the for loop
                    attn_vctr_tmp = copy.deepcopy(attn_vctr)
                    context_tmp = copy.deepcopy(context)
                    hidden_tmp = copy.deepcopy(hidden)
                    log_probs_new = copy.deepcopy(log_probs)
                    # Now add the words to the corresponding sentences 
                    # First, keep only the sentences that remain:
                    sentences = [copy.deepcopy(sentences[parents[0][0][i]]) for i in parents[0][0]]
                    sentences_tmp = copy.deepcopy(sentences)
                    completed_sentences = 0 # Keep a track of how many EOS tokens have appeared
                    for i in range(beam_size):
                        if tokens[0][0][i].item() == 3: # Unk replacement
                            sentences[i].append(src_sent[torch.argmax(attn_vctr[i]).item()]) 
                        else:
                            sentences[i].append(self.vocab.tgt.id2word[tokens[0][0][i].item()])
                        if '</s>' in sentences[i]:
                            sentences[i].remove('</s>') 
                            hypo.append(copy.deepcopy(sentences[i]))
                            sentences_tmp.pop(i) 
                            final_scores.append(log_probs[0][0][i].item())
                            log_probs_new = log_probs_new
                            completed_sentences += 1









                

                

                

                
                
        return hypotheses

    def attention(self, enc_output, decoder_hidden_state, mask, style):
        # Take an argument telling whether you're going with dot prod or MLP
        # For dot prod 
        enc_output = mask.unsqueeze(2)*enc_output # Mask the stuff corresponding to <pad> token
        if style == 1:
            # enc_output: seq_len_src x batch_size x hidden_size 
            # decoder_hidden_state: seq_len_tgt x batch_size x hidden_size (if input feeding, seq_len_enc = 1)
            attn_score = torch.bmm(enc_output.permute(1, 0, 2), decoder_hidden_state.permute(1, 2, 0)) 
            # attn_score: batch_size x seq_len_src x seq_len_tgt 
            attn = self.sm(attn_score)
            # attn:  batch_size x seq_len_src x seq_len_tgt
            ctxt = torch.bmm(enc_output.permute(1, 2, 0), attn)
            # ctxt: batch_size x hidden_size x seq_len_tgt
            ctxt = ctxt.permute(2, 0, 1)
            # ctxt: seq_len_tgt x batch_size x hidden_size
        else: # For MLP
            # enc_output: seq_len_src x batch_size x hidden_size
            # decoder_hidden_state: 1 x batch_size x hidden_size (input feeding)

            # Repeat decoder_hidden_state in order to do concatenation
            seq_len_src = enc_output.shape[0]
            decoder_hidden_state = decoder_hidden_state.repeat(seq_len_src, 1, 1) # seq_len_src x batch_size x hidden_size

            concat = torch.cat((enc_output, decoder_hidden_state), dim = 2) # concat: seq_len_src x batch_size x (2*hidden_size)
            lin1 = self.MLPLin1(self.drop(concat)) # lin1: seq_len_src x batch_size x hidden_size
            lin1_act = self.tanh(lin1) # Activation by tanh
            attn_score = self.MLPLin2(lin1_act) # attn_score: seq_len_src x batch_size x 1
            attn_score = attn_score.permute(1, 0, 2) # attn_score: batch_size x seq_len_src x 1
            attn = self.sm(attn_score) # attn: batch_size x seq_len_src x 1
            ctxt = torch.bmm(enc_output.permute(1, 2, 0), attn) # ctxt: batch_size x hidden_size x 1
            ctxt = ctxt.permute(2, 0, 1) # ctxt: 1 x batch_size x hidden_size

        return ctxt, attn

    def evaluate_ppl(self, dev_data,criterion, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():
            self.eval()
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                src_sents = pad_func(src_sents, '<pad>')
                tgt_sents = pad_func(tgt_sents, '<pad>')
                        
                op, target = self.__call__(src_sents, tgt_sents)
                target= target[1:]
                loss = criterion(op.view(-1, len(self.vocab.tgt)), target.cuda().reshape(-1))
                cum_loss += loss.item()
                
                cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss/ cum_tgt_words)
        self.train()

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, path)
        # raise NotImplementedError()

def pad_func(input, padding):
    #  Takes batch and pads it with <pad>
    l = max(len(sen) for sen in input)
    padded_ip = [sen + [padding]*(l - len(sen)) for sen in input]
    return padded_ip


def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score



def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    print (len(train_data))
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    lr = float(args['--lr'])
    model_save_path = args['--save-to']
    attn_style = args['--attn-style']
    path_to_pretrained = args['--pretrained-model']
    weight_decay = float(args['--weight-decay'])

    vocab = pickle.load(open(args['--vocab'], 'rb'))
    
    if path_to_pretrained != 'None':
        model = torch.load(path_to_pretrained)

    else:
        model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),vocab=vocab, attn_style = attn_style,
                dropout_rate=float(args['--dropout']))
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= weight_decay)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    if path_to_pretrained != 'None':

        print('Reloading pre-trained validation score ...', file=sys.stderr)

        # compute dev. ppl and bleu
        dev_ppl = model.evaluate_ppl(dev_data, criterion, batch_size=128)   # dev batch size can be a bit larger
        valid_metric = -dev_ppl

        print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
        hist_valid_scores.append(valid_metric)

    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            # Pad src_sents
            src_sents = pad_func(src_sents, '<pad>')
            tgt_sents = pad_func(tgt_sents, '<pad>')
            batch_size = len(src_sents)
            # print (batch_size)
            # (batch_size)
            optimizer.zero_grad()
            op, target = model(src_sents, tgt_sents)
            target = target[1:]
            # print (op.shape, target.shape)
            fullloss = criterion(op.view(-1, len(vocab.tgt)), target.cuda().reshape(-1))
            loss = fullloss/tgt_words_num_to_predict
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            report_loss += fullloss.item()
            cum_loss += fullloss.item()

              
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss/ cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, criterion, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                is_equal = valid_metric == max(hist_valid_scores)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    checkpoint = {'optimizer' : optimizer.state_dict()}
                    torch.save(checkpoint, 'checkpoint.pth')

                    # You may also save the optimizer's state
                elif is_equal:
                    lr = lr * float(args['--lr-decay'])
                    print('load previously best model, decay learning rate to %f, and reconstruct optimiser' % lr, file=sys.stderr)
                    model.load(model_save_path) 
                    optimizer = optim.Adam(model.parameters(), lr = lr)

                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= weight_decay)
                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int):
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])
    print (model.attn_style)
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
