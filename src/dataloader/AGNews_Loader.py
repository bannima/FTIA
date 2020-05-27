#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: AGNews_Loader.py
Description: 
Author: Barry Chow
Date: 2019/2/2 11:37 PM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
import torch.nn as nn
from .preprocess import clean_str
import random
from torch.utils.data import Dataset,DataLoader

class AGNewsDataset(object):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set=True
        self.occupy=occupy
        self.contents = []
        self.labels = []
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_seq_len = -1
        self.corpus = []
        filepath = './dataset/AGNews/'

        self.train_dataset,self.train_labels = self.loadDataLabel(filepath+'train_texts.txt',filepath +'train_labels.txt')
        self.test_dataset,self.test_labels = self.loadDataLabel(filepath+'test_texts.txt',filepath+'test_labels.txt')

        #gernerate vocab
        for content in self.train_dataset+self.test_dataset:
            self.corpus.append(content.split())
            for word in content.split():
                if word in self.vocab:
                    self.vocab[word]+=1
                else:
                    self.vocab[word]=1

        #generate word2idx and idx2word
        idx = 0
        for word in self.vocab.keys():
            self.word2idx[word]=idx
            self.idx2word[idx]=word
            idx +=1

        #gernerate label2idx and idx2label
        idx = 0
        for label in set(self.train_labels+self.test_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
            idx +=1

    def loadDataLabel(self,datafile,labelfile):
        retData = []
        retLabel = []
        for line in open(datafile):
            retData.append(line.strip(' \r\n'))
        for line in open(labelfile):
            retLabel.append(line.strip(' \r\n'))
        return retData,retLabel

    def get_vocab(self):
        return self.vocab

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def get_label2idx(self):
        return self.label2idx

    def get_idx2label(self):
        return self.idx2label

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.corpus

    def get_trainset(self):
        return self.train_dataset,self.train_labels

    def get_testset(self):
        return self.test_dataset,self.test_labels

    def get_max_seq_len(self):
        return self.max_seq_len

    #signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(AGNewsDataset, "_instance"):
            AGNewsDataset._instance = AGNewsDataset(*args, **kwargs)
        return AGNewsDataset._instance

class AGNewsDataLoader(Dataset):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = is_train_set
        self.occupy = occupy
        self.agnewsDataset = AGNewsDataset.instance(self.is_train_set,self.occupy)
        if self.is_train_set:
            self.dataset,self.labels = self.agnewsDataset.get_trainset()
        else:
            self.dataset,self.labels = self.agnewsDataset.get_testset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def get_vocab(self):
        return self.agnewsDataset.get_vocab()

    def get_word2idx(self):
        return self.agnewsDataset.get_word2idx()

    def get_label2idx(self):
        return self.agnewsDataset.get_label2idx()

    def get_idx2word(self):
        return self.agnewsDataset.get_idx2word()

    def get_idx2label(self):
        return self.agnewsDataset.get_idx2label()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.agnewsDataset.get_corpus()

    def get_max_seq_len(self):
        return self.agnewsDataset.get_max_seq_len()
