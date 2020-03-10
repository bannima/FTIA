#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: static_vectors.py
Description: static word vectors
Author: Barry Chow
Date: 2019/3/13 9:36 PM
Version: 0.1
"""

import torch
import os

class StaticWordVectors():
    def __init__(self,word2idx,type='6B',dim=300,kind ='glove'):
        self.word2vec = torch.zeros(len(word2idx.keys()),dim)
        path = './wordvectors/.static_vector_cache/'
        filename = kind+'.'+type+'.'+str(dim)+'d.txt'

        count = 0
        for line in open(path+filename):
            word = line.split(' ')[0]
            vector = [float(value) for value in line.split(' ')[1:]]
            if word2idx.get(word):
                self.word2vec[word2idx[word]]=torch.tensor(vector)
                count+=1

        print(count,len(word2idx.keys()))

    def get_wordVectors(self):
        return self.word2vec




