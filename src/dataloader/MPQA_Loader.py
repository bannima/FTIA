#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: MPQA_Loader.py
Description: 
Author: Barry Chow
Date: 2019/2/2 12:30 PM
Version: 0.1
"""
from __future__ import print_function
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import random
import torch
import torch.nn as nn
from preprocess import clean_str
from torch.utils.data import Dataset,DataLoader

class MPQA_Dataset(object):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = is_train_set
        self.occupy=occupy
        filename = './dataset/MPQA/mpqa.all.txt'
        self.contents = []
        self.labels = []
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
        self.max_seq_len = -1
        self.corpus = []
        # load dataset
        for line in open(filename):
            label = line.split()[0]
            self.labels.append(label)
            #skip those samples with no content
            if len(line.split())<=1:
                continue
            content = clean_str(' '.join(line.split()[1:]))
            self.corpus.append(content.split(' '))
            if len(content.split()) > self.max_seq_len:
                self.max_seq_len = len(content.split())
            self.contents.append([content, label])
        self.len = len(self.contents)
        # generate vocab
        for content in self.contents:
            for word in content[0].split():
                if word in self.vocab:
                    self.vocab[word] += 1
                else:
                    self.vocab[word] = 1
        # generate word2idx and idx2word
        idx = 0
        for word in self.vocab.keys():
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        # generate label2idx and idx2label
        idx = 0
        for label in set(self.labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
            idx += 1

        # split train and test dataset
        random.shuffle(self.contents)
        self.trainset = self.contents[:int(self.occupy * self.len)]
        self.testset = self.contents[int(self.len * self.occupy):]

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
        if not hasattr(MPQA_Dataset, "_instance"):
            MPQA_Dataset._instance = MPQA_Dataset(*args, **kwargs)
        return MPQA_Dataset._instance

class MPQADataLoader(Dataset):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = True
        self.occupy = occupy
        self.mpqa_dataset = MPQA_Dataset.instance(self.is_train_set,self.occupy)
        if self.is_train_set:
            self.dataset = self.mpqa_dataset.get_trainset()
        else:
            self.dataset = self.mpqa_dataset.get_testset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]

    def get_vocab(self):
        return self.mpqa_dataset.get_vocab()

    def get_word2idx(self):
        return self.mpqa_dataset.get_word2idx()

    def get_label2idx(self):
        return self.mpqa_dataset.get_label2idx()

    def get_idx2word(self):
        return self.mpqa_dataset.get_idx2word()

    def get_idx2label(self):
        return self.mpqa_dataset.get_idx2label()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.mpqa_dataset.get_corpus()

    def get_max_seq_len(self):
        return self.mpqa_dataset.get_max_seq_len()



