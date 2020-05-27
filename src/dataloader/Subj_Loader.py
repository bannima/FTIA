#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: Subj_Loader.py
Description: 
Author: Barry Chow
Date: 2019/2/2 10:38 AM
Version: 0.1
"""
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import random
import torch
import torch.nn as nn
from .preprocess import clean_str
from torch.utils.data import Dataset,DataLoader

class Subj_dataset(object):
    def __init__(self,is_train_set=True,occpy=0.7):
        '''
        load dataset and split train and test
        :param is_train_set:
        :param occpy:
        '''
        filename ='./dataset/Subj/subj.all.txt'
        self.contents = []
        self.labels = []
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_seq_len = -1
        self.is_train_set = is_train_set
        self.occupy = occpy
        self.corpus = []
        #load dataset
        for line in open(filename,encoding='utf-16'):
            label = line.split()[0]
            self.labels.append(label)
            content = clean_str(' '.join(line.split()[1:]))
            self.corpus.append(content.split(' '))
            if len(content.split())>self.max_seq_len:
                self.max_seq_len=  len(content.split())
            self.contents.append([content,label])
        self.len = len(self.contents)
        #generate vocab
        for content in self.contents:
            for word in content[0].split():
                if word in self.vocab:
                    self.vocab[word]+=1
                else:
                    self.vocab[word]=1

        #generate word2idx and idx2word
        idx = 0
        for word in self.vocab.keys():
            self.word2idx[word]=idx
            self.idx2word[idx]=word
            idx+=1

        #generate label2idx and idx2label
        idx = 0
        for label in set(self.labels):
            self.label2idx[label]=idx
            self.idx2label[idx]=label
            idx +=1

        #split train and test dataset
        random.shuffle(self.contents)
        self.trainset = self.contents[:int(self.occupy*self.len)]
        self.testset = self.contents[int(self.len*self.occupy):]

        print(' num of words in vocabulary ', len(self.word2idx.keys()))
        print(' num of samples in train dataset', len(self.trainset))
        print(' num of samples in test dataset', len(self.testset))
        print(' num of samples in all dataset', len(self.trainset) + len(self.testset))


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

    def get_trainset(self):
        return self.trainset

    def get_testset(self):
        return self.testset

    def get_corpus(self):
        return self.corpus

    def get_max_seq_len(self):
        return self.max_seq_len

    # signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(Subj_dataset, "_instance"):
            Subj_dataset._instance = Subj_dataset(*args, **kwargs)
        return Subj_dataset._instance

class SubjDatasetLoader(Dataset):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = is_train_set
        self.occupy = occupy
        self.subjDataset = Subj_dataset.instance(self.is_train_set,self.occupy)
        if self.is_train_set:
            self.dataset = self.subjDataset.get_trainset()
        else:
            self.dataset = self.subjDataset.get_testset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]

    def get_vocab(self):
        return self.subjDataset.get_vocab()

    def get_word2idx(self):
        return self.subjDataset.get_word2idx()

    def get_label2idx(self):
        return self.subjDataset.get_label2idx()

    def get_idx2word(self):
        return self.subjDataset.get_idx2word()

    def get_idx2label(self):
        return self.subjDataset.get_idx2label()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.subjDataset.get_corpus()
    
    def get_max_seq_len(self):
        return self.subjDataset.get_max_seq_len()


