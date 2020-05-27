#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: TwentyNews_Loader.py
Description: 
Author: Barry Chow
Date: 2019/1/30 6:53 PM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import random
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import os
from .preprocess import clean_str

class TwentyNewsDataset(object):
    def __init__(self,is_train_set=True,occupy = 0.7):
        filepath = './dataset/20News/'
        self.contents =[]
        self.labels = []
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_seq_len = -1
        self.is_train_set = is_train_set
        self.corpus = []
        #load dataset
        for dir in os.listdir(filepath):
            self.labels.append(dir)
            for file in os.listdir(filepath+dir):
                f = open(filepath+dir+'/'+file)
                self.contents.append([clean_str(f.read()),dir])
        #generate vocab
        for content in self.contents:
            self.corpus.append(content[0])
            #get max seq len
            if len(content[0].split())>self.max_seq_len:
                self.max_seq_len = len(content[0].split())
            for word in content[0].split():
                if word in self.vocab:
                    self.vocab[word]+=1
                else:
                    self.vocab[word]=1
        self.len = len(self.contents)
        #generate word2idx,idx2word
        idx = 0
        for word in self.vocab.keys():
            self.word2idx[word]=idx
            self.idx2word[idx]=word
            idx +=1
        #generate label2idx and idx2label
        idx = 0
        for label in set(self.labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
            idx +=1

        #split train and test dataset
        random.shuffle(self.contents)
        self.train_dataset =self.contents[:int(self.len*occupy)]
        self.test_dataset = self.contents[int(self.len*occupy):]

    def get_trainset(self):
        return self.train_dataset

    def get_testset(self):
        return self.test_dataset

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def get_vocab(self):
        return self.vocab

    def get_label2idx(self):
        return self.label2idx

    def get_idx2label(self):
        return self.idx2label

    def get_corpus(self):
        return self.corpus

    def get_max_seq_len(self):
        return self.max_seq_len

    #signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(TwentyNewsDataset, "_instance"):
            TwentyNewsDataset._instance = TwentyNewsDataset(*args, **kwargs)
        return TwentyNewsDataset._instance

class TwentyNewsDataLoader(Dataset):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = is_train_set
        self.twentyNewsLoader = TwentyNewsDataset.instance(self.is_train_set, occupy)
        if self.is_train_set:
            self.dataset = self.twentyNewsLoader.get_trainset()
        else:
            self.dataset = self.twentyNewsLoader.get_testset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]

    def get_word2idx(self):
        return self.twentyNewsLoader.get_word2idx()

    def get_vocab(self):
        return self.twentyNewsLoader.get_vocab()

    def get_label2idx(self):
        return self.twentyNewsLoader.get_label2idx()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.twentyNewsLoader.get_corpus()

    def get_max_seq_len(self):
        return self.twentyNewsLoader.get_max_seq_len()


