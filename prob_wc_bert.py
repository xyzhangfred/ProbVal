#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:35:01 2019

@author: xiongyi
"""
from __future__ import absolute_import, division, unicode_literals
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#




import os
import io
import copy
import logging
import numpy as np
from senteval.probing import PROBINGEval
from senteval.tools.validation import SplitClassifier

import torch

class PROBINGEvalWithKey(PROBINGEval):
    def __init__(self, task, task_path, seed=1111):
        self.seed = seed
        self.task = task
        logging.debug('***** (Probing) Transfer task : %s classification *****', self.task.upper())
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}
        self.loadFile(task_path)
        logging.info('Loaded %s train - %s dev - %s test for %s' %
                     (len(self.task_data['train']['y']), len(self.task_data['dev']['y']),
                      len(self.task_data['test']['y']), self.task))

    def run_with_tok(self, params, batcher):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size
        logging.info('Computing embeddings for train/dev/test')
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.task_data[key]['X'], self.task_data[key]['y'] = map(list, zip(*sorted_data))
            self.task_data[key]['y_X'] = map(list, zip(*sorted_data))
            task_embed[key]['X'] = []
            for ii in range(0, len(self.task_data[key]['y']), bsize):
                batch_with_key = self.task_data[key]['y_X'][ii:ii + bsize]
                
                embeddings = batcher(params, batch_with_key)
                task_embed[key]['X'].append(embeddings)
            task_embed[key]['X'] = np.vstack(task_embed[key]['X'])
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
        logging.info('Computed embeddings')

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        if self.task == "WordContent" and params.classifier['nhid'] > 0:
            config_classifier = copy.deepcopy(config_classifier)
            config_classifier['classifier']['nhid'] = 0 
            print(params.classifier['nhid'])

        clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['dev']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['dev']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, self.task.upper()))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(task_embed['dev']['X']),
                'ntest': len(task_embed['test']['X'])}

class WordContentEval(PROBINGEvalWithKey):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'word_content.txt')
        # labels: 200 target words
        PROBINGEvalWithKey.__init__(self, 'WordContent', task_path, seed)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

#        label_id = label_map[example.label]
#        if ex_index < 5:
#            logger.info("*** Example ***")
#            logger.info("guid: %s" % (example.guid))
#            logger.info("tokens: %s" % " ".join(
#                    [str(x) for x in tokens]))
#            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#            logger.info(
#                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features
       
        
def batcher(params, batch):
    #print ('batch size' ,len(batch))
    batch = [[dat[0],dat[1]] for dat in batch if dat != [] ]
    #print ('batch', batch)
    examples = []
    unique_id = 0
    #print ('batch size ', len(batch))
    for dat in batch:
        sent = dat[1].strip()
        text_b = None
        text_a = sent
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1

    features = convert_examples_to_features(examples, params['bert'].seq_length, params['bert'].tokenizer)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    
    all_encoder_layers, _ = params['bert'](all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
    
    
    ###get z_vec
    #get the output of previous layer
    prev_out = all_encoder_layers[params['bert'].layer_no -1] 
    #print ('prev_out.shape ', prev_out.shape)
    #print ('all_input_mask.shape ', all_input_mask.shape)
    ##apply self-attention to it
    
    
    extended_attention_mask = all_input_mask.cuda().unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(params['bert'].parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    embeddings = next(params['bert'].children()).encoder.layer[params['bert'].layer_no].attention.self(prev_out, extended_attention_mask)
    ##do mean/max pooling
    
    
    embeddings = embeddings.detach().mean(1).cpu().numpy()
    #print ('befor shape', embeddings.shape)
    if params['bert'].head_no is not None:
        if params['bert'].head_no == 'random':
            embeddings = embeddings[:, params['bert'].randidx]
        else:
            embeddings = embeddings[:,64 * params['bert'].head_no : 64 * (params['bert'].head_no +1)]
    #print ('after shape', embeddings.shape)

    #print ('embeddings.shape ', embeddings.shape)
    #print ('finished a batch \n\n')

    return embeddings


























