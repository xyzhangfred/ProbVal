#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:33:55 2019

@author: xiongyi
"""

from __future__ import absolute_import, division, unicode_literals

import sys,os
import io
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG, filename = 'TraceWord')
#logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
#logging.info('test')


import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import argparse
# Set PATHs
PATH_TO_SENTEVAL = './senteval'
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL,'../data')


# PATH_TO_VEC = 'glove/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set up logger
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 2}
params_senteval['classifier'] = {'nhid': 100, 'optim': 'adam', 'batch_size': 32,
                             'tenacity': 3, 'epoch_size': 4, 'kfold':3}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, text_a, text_b=None):
        """Constructs a InputExample.

        Args:
            unique_id: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
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

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        #print ('tokens', tokens)
        #print ('input_ids', input_ids)
        

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

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features
     

###new task!

# SentEval prepare and batcher
def prepare(params, samples):
    params.batch_size = 128
    #print ('samples.shape', samples.shape)
    return

def batcher(params, batch, labels):
    #print ('batch[0]' ,batch[0])

    #assert len(batch[0]) == 2, 'batch format error'
    batch = [ ' '.join(dat) for dat in batch if dat != [] ]
    #print ('batch size' ,len(batch[0]))
    #print ('batch[0]' ,batch[0])
    #print ('batch', batch)
    examples = []
    unique_id = 0
    #print ('batch size ', len(batch))
    for dat in batch:
        sent = dat.strip()
        text_b = None
        text_a = sent
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1
        
    
    features = convert_examples_to_features(examples, params['bert'].seq_length, params['bert'].tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(params['bert'].device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(params['bert'].device)
     
    ###get z_vec.detach().cpu().numpy()
    #get the output of previous layer
    if params['bert'].layer_no > 0:
        all_encoder_layers, _ = params['bert'](all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
        prev_out = all_encoder_layers[params['bert'].layer_no -1]
    else:
        #to get representation in layer 0, we need to get the output of the embedding layer of bert.
        prev_out = params_senteval['bert'].embeddings(all_input_ids, token_type_ids=None)
    #print ('prev_out.shape ', prev_out.shape)
    #print ('all_input_mask.shape ', all_input_mask.shape)
    ##apply self-attention to it
    
    extended_attention_mask = all_input_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(params['bert'].parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    embeddings = params_senteval['bert'].encoder.layer[params['bert'].layer_no].attention.self.value(prev_out)
    embeddings = embeddings.detach().cpu().numpy()
    
    if params['bert'].head_no is not None:
        if params['bert'].head_no == 'random':
            embeddings = embeddings[:,:,params['bert'].randidx]
        else:
            embeddings = embeddings[:,:,64 * params['bert'].head_no : 64 * (params['bert'].head_no +1)]
    #print ('befor shape', embeddings.shape)
    word_embedding = params_senteval['bert'].embeddings(all_input_ids, token_type_ids=None).detach().cpu().numpy()
    new_emb = []
    for i in range(len(batch)):
        sent_len = sum(all_input_mask[i])
        if sent_len < 2:
            continue
        if labels[i] == 1:
            t = np.random.randint(1,sent_len)
            w_e = word_embedding[i,t,:]
            c_e = embeddings[i,t,:]
            
            token_emb = np.concatenate((w_e, c_e), axis = 0)
            #print ('token_emb shape', token_emb.shape)
            new_emb.append(token_emb)
        else:
            t = np.random.randint(1,sent_len)
            w_e = word_embedding[i,t,:]
            not_t = np.random.choice([x for x in range(sent_len) if x != t],size = 1)[0]
            c_e = embeddings[i,not_t,:]
            token_emb = np.concatenate((w_e, c_e), axis = 0)
            #print ('token_emb shape', token_emb.shape)
            new_emb.append(token_emb)
    
    ###concatenate back with the original word embedding of the word!            
    #print ('word_embedding shape', word_embedding.shape)
    #print ('np.asanyarray(new_emb) shape', np.asanyarray(new_emb).shape)
    embeddings = np.asanyarray(new_emb)
    #print ('befor shape', embeddings.shape)
    #print ('after shape', embeddinglayer_nos.shape)

    #print ('embeddings.shape ', embeddings.shape)
    #print ('finished a batch \n\n')

    return embeddings



def main(head_no = None, layer_no = -1):
    
    logging.info('\nDoing head number '+str(head_no) + '!\n')

    
    params_senteval['bert'].layer_no = layer_no
    params_senteval['bert'].head_no = head_no
    
    params_senteval['bert'].tokenizer = tokenizer
    params_senteval['bert'].device = device
    params_senteval['bert'].randidx = np.random.choice(np.arange(768), size = 64, replace=False)
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['TraceWord']
    results = se.eval(transfer_tasks)
    print('results for head ', head_no, ' layer ', layer_no, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--layer_no", default= -2, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for predictions.")

    args = parser.parse_args(['--local_rank', '1', '--do_lower_case'])
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

#    if args.local_rank != -1:
#        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                          output_device=args.local_rank)
#    elif n_gpu > 1:
#        model = torch.nn.DataParallel(model)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        params_senteval['bert'] = next(model.children())
    else:
        params_senteval['bert'] = model
    params_senteval['bert'].seq_length = args.max_seq_length

    model.eval()
    for layer_no in range(0,12):
        main(head_no = None, layer_no = layer_no)
        main(head_no = 'random', layer_no = layer_no)
        for head_no in range(12):
            main(head_no, layer_no)
        

    #for layer_no in range(1):
       
    
    
    
    
    
    
    
    
    
    
    
