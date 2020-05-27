#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: FTIA.py
Description: Fusion of Background Information based Attention Mechanism
Author: Barry Chow
Date: 2019/1/29 6:28 PM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.MovieReview_Loader import MovieReviewDataset
from tools import create_variable
import torch.nn.functional as f
from wordvectors.static_vectors import StaticWordVectors
from torch.autograd.variable import Variable
from tools import saveTextRepresentations,save_attention_weights

class FTIA(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,n_layers,word_vector=None,bidirectional=False,fine_tuned=False,datasetType=None):
        super(FTIA, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.output_size = output_size
        self.word_embedding = nn.Embedding(input_size, hidden_size)

        if word_vector is not None:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=fine_tuned)
        self.label_embedding = nn.Embedding(output_size,hidden_size*self.n_directions)
        self.label_embedding.weight = nn.Parameter(create_variable(torch.randn(output_size,hidden_size*self.n_directions)),requires_grad=True)

        self.mp = nn.MaxPool2d((1,self.output_size))
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,bidirectional=bidirectional)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)
        self.datasetType = datasetType

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)

    def forward(self, input, seq_lengths, labels):
        #preprocess label embedding
        labels_cuda = create_variable(torch.LongTensor([range(self.output_size) for label in labels]))
        input = input.t()
        self.batch_size = input.size(1)
        cell = self._init_hidden(self.batch_size)
        hidden = self._init_hidden(self.batch_size)

        # word embedding
        #print input

        word_embedded = self.word_embedding(input)

        # label embedding
        label_embedded = self.label_embedding(labels_cuda)
        self.lstm.flatten_parameters()
        # to compact weights again call flatten paramters

        output, (final_hidden_state, final_cell_state) = self.lstm(word_embedded, (hidden, cell))

        # attention layers
        att_output, weights = self.label_attention(output, label_embedded)
        #save text representations
        saveTextRepresentations(self.datasetType,'FTIA',att_output.detach().cpu().numpy())

        #penalty
        penalty = self.calc_penalty(weights)
        return self.fc(att_output),penalty

    def label_attention(self, output, label_embedding):
        '''
        calculate attention representations for text
        :param output:
        :param label_embedding:
        :return: attentioned representations
        '''
        output = output.permute(1,2,0)

        #l2 norm weights
        label_embedding = f.normalize(label_embedding,dim=2,p=2)
        output = f.normalize(output,dim=1,p=2)

        weights = torch.bmm(label_embedding,output)

        # change BXOXS to BXOXS
        weights = weights.permute(0, 2, 1)

        #max pooling to BXSX1
        weights = self.mp(weights)
        weights = F.softmax(weights,dim=1)

        #save weights
        #save_attention_weights(self.datasetType,'FTIA',weights.squeeze(2).detach().numpy())

        # BXIXS * BXSX1 = BXIX1
        weighted_output = torch.bmm(output, weights)

        return weighted_output.squeeze(2), weights

    def calc_penalty(self,weights):
        return Variable(torch.log(1/torch.sum(torch.var(torch.tensor(weights)),dim=0)),requires_grad=True)





