#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: SelfAttention.py
Description: 
Author: Barry Chow
Date: 2019/3/2 10:15 AM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tools import create_variable,saveTextRepresentations
from torch.nn import functional as F
from torch.autograd.variable import Variable
from tools import save_attention_weights

class SelfAttentionClassifier(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers,bidirectional=False,word_vector=None,datasetType=None):
        super(SelfAttentionClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.n_directions = int(bidirectional)+1

        self.word_embedding = nn.Embedding(input_size, hidden_size)
        if word_vector:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=False)

        self.lstm = nn.LSTM(hidden_size,hidden_size,n_layers,bidirectional)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        da = 350
        r = 30
        penalization_coeff = 1.0
        self.W_s1 = nn.Linear(self.n_directions*hidden_size,da)
        self.W_s2 = nn.Linear(da,r)
        self.fc = nn.Linear(r*self.n_directions*hidden_size,2000)
        self.label = nn.Linear(2000,output_size)
        self.datasetType = datasetType


    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)

    def forward(self,input,seq_lengths,labels):
        global global_datasetType
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        cell = self._init_hidden(batch_size)
        # embedding SXB to SXBXI
        embedd = self.word_embedding(input)

        # pack them up nicely
        lstm_input = pack_padded_sequence(
            embedd, seq_lengths.data.cpu().numpy()
        )

        # to compact weights again call flatten paramters
        output, (final_hidden_state, final_cell_state) = self.lstm(lstm_input, (hidden, cell))

        output,_ = pad_packed_sequence(output,batch_first=True)

        #output = output.permute(1,0,2)

        attention_weight_matrix = self.attention_net(output)

        # save weights
        save_attention_weights(self.datasetType,'SelfAtt',torch.sum(attention_weight_matrix,dim=1).detach().numpy())

        penalty = self.calc_penalty(attention_weight_matrix)
        hidden_matrix = torch.bmm(attention_weight_matrix,output)


        fc_output = self.fc(hidden_matrix.view(-1,hidden_matrix.size()[1]*hidden_matrix.size()[2]))

        logits = self.label(fc_output)

        return logits,penalty


    def attention_net(self,lstm_output):
        '''
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
		encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
		the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
		connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
		pos & neg.
        :param lstm_output:
        :return:
        '''

        attention_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attention_weight_matrix = attention_weight_matrix.permute(0,2,1)
        attention_weight_matrix = F.softmax(attention_weight_matrix,dim=2)

        return attention_weight_matrix

    def calc_penalty(self,attention_weight_matrix):
        dim = attention_weight_matrix.shape[1]
        A_T = attention_weight_matrix.permute(0,2,1)
        penalty = torch.norm(torch.bmm(attention_weight_matrix,A_T)-torch.eye(1),p='fro',dim=[1,2])
        return Variable(torch.mean(penalty),requires_grad=True)
        #return penalty

