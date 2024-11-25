### Basic logistical, wrangling, & visualization imports

import os
import glob
import torch

### Model imports
from transformers import BertModel, BertConfig
import TokenizerChanger as tc
import numpy

### Import the necessary

import torch
from transformers import BertForMaskedLM, BertTokenizer

### Create the tokenizer and the model

tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model.eval()



print(len(tokenizer.vocab))

token_dict = []

with open('end_tokens.txt', mode='r', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        token = line[1]
        if token not in token_dict:
            token_dict += [token]
            del tokenizer.vocab[token]

# print(len(tokenizer.vocab))

with open('nounlist_single-token-plurals.csv', mode='r', encoding='utf-8') as f:
     for line in f:
        if line[0] != ',':
            line = line.split(',')
            token = line[1]
            if token in tokenizer.vocab.keys():
                del tokenizer.vocab[token]

print(len(tokenizer.vocab))

new_tokens_multi = []
with open('nounlist_multi-token-nonmorph-plurals.csv', mode='r', encoding='utf-8') as f:
    for line in f:
        if line[0] != ',':
            line = line.split(',')
            word = line[1]
            new_tokens_multi.append((word, tokenizer.convert_ids_to_tokens(tokenizer.encode(word))))

with open('new_tokenizations_multi-nonmorph.txt', mode='w', encoding='utf-8') as f:
        for item in new_tokens_multi:
            tokens = item[1]
            token_list = '(' + ','.join(tokens) + ')'
            f.write(item[0] + ', ' + token_list + '\n')

new_tokens_single = []

with open('nounlist_single-token-plurals.csv', mode='r', encoding='utf-8') as f:
    for line in f:
        if line[0] != ',':
            line = line.split(',')
            word = line[1]
            new_tokens_single.append((word, tokenizer.convert_ids_to_tokens(tokenizer.encode(word))))

with open('new_tokenizations_single.txt', mode='w', encoding='utf-8') as f:
        for item in new_tokens_single:
            tokens = item[1]
            token_list = '(' + ','.join(tokens) + ')'
            f.write(item[0] + ', ' + token_list + '\n')
