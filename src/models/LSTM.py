#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: LSTM.py
Description: 
Author: Barry Chow
Date: 2019/1/29 7:07 PM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.nn import functional as F
from tools import create_variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tools import saveTextRepresentations

class LSTMClassifier(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,
                 n_layers,bidirectional=True,word_vector= None,fine_tuned=False):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional)+1

        #with pertrianed word vectors or random initialize
        self.word_embedding = nn.Embedding(input_size, hidden_size)
        if word_vector is not None:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=fine_tuned)

        self.lstm = nn.LSTM(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size,output_size)

    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,
                             batch_size,self.hidden_size)
        return create_variable(hidden)


    def forward(self,input,seq_lengths,labels):
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        cell = self._init_hidden(batch_size)
        # embedding SXB to SXBXI
        embedd = self.word_embedding(input)

        # pack them up nicely
        lstm_input =pack_padded_sequence(
            embedd, seq_lengths.data.cpu().numpy()
        )

        # to compact weights again call flatten paramters
        output,(final_hidden_state,final_cell_state) = self.lstm(lstm_input,(hidden,cell))

        # save text representations
        #saveTextRepresentations('TREC', 'LSTM', output.detach().numpy())

        #no penalty
        return self.fc(final_hidden_state[-1]),0.0




