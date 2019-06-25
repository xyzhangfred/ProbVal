#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:50:29 2019

@author: xiongyi
"""
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer


file_path = './data/word_content.txt'

key_nos = []
key_words = []

with (open (file_path, 'r')) as f:
    lines = f.readlines()
    for line in lines:
        data = line.split('\t')
        tag = data[0]
        key = data[1]
        sent = data[-1]
        sent_words = sent.split(' ')
        key_no = [i for i in range(len(sent_words)) if sent_words[i].lower() == key]
        key_nos.append(key_no)
        key_words.append(key)

for i,kn in enumerate(key_nos):
    if len(kn) != 1:
        print (str(i) +' th has length ', str(len(kn)))
        
    
##seems like word piece embedding is not an issue, except for "t-shirt"
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#for kw in key_words:
#    tokens = tokenizer.wordpiece_tokenizer.tokenize(kw)
#    if len(tokens) > 1:
#        print (kw)

new_file_path = './data/probing/word_content_addw.txt'


with open (file_path, 'r') as f:
    with open (new_file_path, 'w') as g: 
        lines = f.readlines()
        for line in lines:
            data = line.split('\t')
            tag = data[0]
            key = data[1]
            newline = data[0] + '\t' + data[1] + '\t' + data[2].strip() + ' ' + data[1] + '\n'
            if key != 't-shirt':
                g.write(newline)
        
        