#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: pretrain_wordvectors.py
Description: pretrain word vectors using word2vec and corpus
Author: Barry Chow
Date: 2019/3/14 2:26 PM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import Word2Vec
import os
import torch

class PretrainedWordVectors():
    def __init__(self,corpus,dataname,word2idx,kind='word2vec',dim=300,):
        path = './wordvectors/.pretrain_vector_cache/'+str(dataname)+'/'
        filename = 'word2vec'+str(dim)+'d.txt'
        self.word2vec = torch.zeros(len(word2idx.keys()),dim)
        if os.path.exists(path+filename):
            model = Word2Vec.load(path+filename)
        else:
            if not os.path.exists(path):
                os.mkdir(path)
            model = Word2Vec(corpus,workers=4,min_count=1,size=dim)
            model.save(path+filename)

        #save weights
        for word in word2idx:
            try:
                vector = torch.tensor(model.wv.word_vec(word))
            except Exception:
                print word
                continue
            self.word2vec[word2idx[word]]=vector

    def get_wordVectors(self):
        return self.word2vec

