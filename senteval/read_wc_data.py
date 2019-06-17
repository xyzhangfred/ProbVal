#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:50:29 2019

@author: xiongyi
"""
import numpy as np

file_path = '../data/probing/word_content.txt'

key_nos = []

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
        

for i,kn in enumerate(key_nos):
    if len(kn) != 1:
        print (str(i) +' th has length ', str(len(kn)))