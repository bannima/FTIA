#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: BiGRU.py
Description: bidirectional GRU model
Author: Barry Chow
Date: 2019/1/29 10:44 AM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tools import create_variable

class BiGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers,bidirectional=True,word_vector=None):
        super(BiGRU,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional)+1
        self.bidirectional = bidirectional

        self.word_embedding = nn.Embedding(input_size, hidden_size)
        if word_vector:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=False)

        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.n_directions,output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,
                             batch_size,self.hidden_size)
        return create_variable(hidden)

    def forward(self,input,seq_lengths,labels):
        #input shape: BXS
        #transpose to SXB
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)

        #embedding SXB to SXBXI
        gru_input = self.word_embedding(input)

        #pack them up nicely
        '''gru_input = pack_padded_sequence(
            embedd,seq_lengths.data.cpu().numpy()
        )'''

        #to compact weights again call flatten paramters
        self.gru.flatten_parameters()
        output,hidden = self.gru(gru_input,hidden)

        #output,_ = pad_packed_sequence(output,batch_first=False)

        #note we use the last hidden state's concat as the input of fc,
        #not the last later output, because there is too much zeros after pad
        #pack operation.

        #output size: Seq X Batch X HIDDEN*2
        fc_output = self.fc(output[-1,:,:])

        #no penalty
        return fc_output,0.0