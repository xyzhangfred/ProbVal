#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:09:03 2019

@author: xiongyi
"""
import logging
import os,sys
import torch
import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity


from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel



#params
device_no = 0
bert_model = 'bert-base-uncased'
max_seq_length = 128
layer_no = -1
head_no = 0
#####load model


device = torch.device("cuda", device_no)
n_gpu = 1
logging.info('\nDoing head number '+str(head_no) + '!\n')
model = BertModel.from_pretrained(bert_model)
model.to(device)
#    if args.local_rank != -1:
#        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                          output_device=args.local_rank)
#    elif n_gpu > 1:
#        model = torch.nn.DataParallel(model)

model.eval()
tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)


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
        #print ('key_word_id', key_word_id)
        #print ('input_ids', input_ids)
        
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
       
    

def get_reps(params, batch):
    #print ('batch[0]' ,batch[0])
    model = params['bert']
    #assert len(batch[0]) == 2, 'batch format error'

    

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
     
    ###get z_vec
    #get the output of previous layer
 
    
    #print ('prev_out.shape ', prev_out.shape)
    #print ('all_input_mask.shape ', all_input_mask.shape)
    ##apply self-attention to it
    
    extended_attention_mask = all_input_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(params['bert'].parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    
     #get the value embeddings of the whole sentence of a certain layer at a certain attention head. 
     #Todo: 1. look at the rank of each head/layer for a sentence, or better, look at the singular values (nuclear norm)
     #2. Look at the average cosine similarity of each head/layer for a sentence
     #3. Cosine similarity to the original embedding of this token. for each tokein in the sentence for each layer
     
    all_encoder_layers, _ = model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)

    num_layer = len(all_encoder_layers)
    num_heads = 12
    rank_z = np.empty((len(batch),num_layer))
    rank_v = np.empty((len(batch),num_layer, num_heads))
    cos_z = np.empty((len(batch),num_layer))
    cos_v = np.empty((len(batch),num_layer, num_heads))
    cos_t = np.empty((len(batch),num_layer, params['bert'].seq_length))
    nuc_norm_z = np.empty((len(batch),num_layer))
    nuc_norm_v = np.empty((len(batch),num_layer, num_heads))
    all_zs = []
    all_vs = []
    for s in range(len(batch)):
        sent_len = len(batch[s].split(' ')) + 2
        for l in range(num_layer):
            if l > 0:
                prev_out = all_encoder_layers[l -1]
            else:
                #to get representation in layer 0, we need to get the output of the embedding layer of bert.
                prev_out = model.embeddings(all_input_ids, token_type_ids=None)
            zs = model.encoder.layer[l].attention.self(prev_out, extended_attention_mask).cpu().detach().numpy()[s,:sent_len,:]
            all_zs.append(zs)
            all_vs.append([])
            rank_z[s,l] = np.linalg.matrix_rank(zs)
            nuc_norm_z[s,l] = np.linalg.norm(zs, ord='nuc')
            cos_dist_mat = 1 - cosine_similarity(zs,zs)
            cos_z[s,l] = np.mean(np.mean(cos_dist_mat))
            for h in range(num_heads):
                vs = model.encoder.layer[l].attention.self.value(prev_out)[s,:sent_len,64 * h : 64 * (h +1)].cpu().detach().numpy()
                all_vs[l].append(vs)
                rank_v[s,l,h] = np.linalg.matrix_rank(vs)
                nuc_norm_v[s,l,h] = np.linalg.norm(vs, ord='nuc')
                cos_dist_mat = 1 - cosine_similarity(vs,vs)
                cos_v[s,l,h] = np.mean(np.mean(cos_dist_mat))
            for t in range(sent_len):
                zt_init = model.embeddings(all_input_ids, token_type_ids=None)[s,t,:].cpu().detach().numpy()
                z_ts = zs[t,:]
                
                cos_t[s,l,t] = scipy.spatial.distance.cosine(z_ts, zt_init)
            
    
     

    #print ('after shape', embeddinglayer_nos.shape)
    #print ('embeddings.shape ', embeddings.shape)
    #print ('finished a batch \n\n')

    return [rank_z,rank_v,cos_z,cos_v,cos_t,nuc_norm_z, nuc_norm_v]


if __name__ == "__main__":

    params = {}
    params['bert'] = model
    params['bert'].device = device
    
    params['bert'].seq_length = max_seq_length
    params['bert'].tokenizer = tokenizer


    corpus_dir = '/media/data/xiongyi/corpus'
    #fns = os.listdir(corpus_dir)
    fn = 'book_100sents.txt'
    
    with open (os.path.join(corpus_dir,fn), 'r') as f:
        lines = f.readlines()
        batch = [l.strip() for l in lines][:10]
    
    results = get_reps(params,batch)

    

















