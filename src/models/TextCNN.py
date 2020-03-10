#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: TextCNN.py
Description: 
Author: Barry Chow
Date: 2019/1/29 7:28 PM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifer(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,word_vector=None):
        super(TextCNNClassifer, self).__init__()
        self.word_embedding = nn.Embedding(input_size, hidden_size)
        if word_vector:
            self.word_embedding.weight = nn.Parameter(word_vector, requires_grad=False)

        self.conv1 = nn.Conv2d(1,10,kernel_size=(5,hidden_size))
        #self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(11570,output_size)

    def forward(self,input,seq_lengths,labels):
        input_size = input.size(0)
        input = self.word_embedding(input)
        #add channel dimension
        input = input.unsqueeze(1)
        x = F.relu(self.mp(self.conv1(input)))
        #x = F.relu(self.mp(self.conv2(x)))
        x = x.view(input_size,-1)#flatten the tensor
        x = self.fc(x)
        #no penalty
        return F.log_softmax(x,dim=1),0.0





