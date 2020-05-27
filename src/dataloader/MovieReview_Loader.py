#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: MovieReview_Loader.py
Description: movie review data loader
Author: Barry Chow
Date: 2019/1/29 10:44 AM
Version: 0.1
"""
from __future__ import print_function
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import torch
from torch.utils.data import DataLoader,Dataset
import random
from .preprocess import clean_str

class MovieReviewDataset(object):
    def __init__(self,is_train_set=True,occupy=0.7):
        filepath = './dataset/MR/'
        pos_filename = filepath + 'rt-polarity.pos'
        neg_filename = filepath + 'rt-polarity.neg'
        reviews = []
        self.is_train_set = is_train_set
        self.train_dataset=[]
        self.test_dataset = []
        self.vocab = {}
        self.word2idx={}
        self.idx2word={}
        self.labelset =set()
        self.label2idx={}
        self.idx2label={}
        self.max_seq_len = -1
        self.corpus = []

        for line in open(pos_filename):
            reviews.append([clean_str(line),'pos'])

        for line in open(neg_filename):
            reviews.append([clean_str(line),'neg'])

        self.len = len(reviews)

        #generate vocab
        for review in reviews:
            words = [word for word in review[0].split()]
            self.corpus.append(words)
            self.labelset.add(review[1])

            if len(words)>self.max_seq_len:
                self.max_seq_len = len(words)

            for word in words:
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

        #generate label2idx and idx2label
        idx = 0
        for label in self.labelset:
            self.label2idx[label] = idx
            self.idx2label[idx] = label
            idx+=1

        #split train and test dataset
        random.shuffle(reviews)
        for index in range(len(reviews)):
            if index <=len(reviews)*occupy:
                self.train_dataset.append(reviews[index])
            else:
                self.test_dataset.append(reviews[index])

        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)

        print(' num of words in vocabulary ', len(self.word2idx.keys()))
        print(' num of samples in train dataset', len(self.train_dataset))
        print(' num of samples in test dataset', len(self.test_dataset))
        print(' num of samples in all dataset',len(self.train_dataset)+len(self.test_dataset))


    def get_idx2word(self):
        return self.idx2word

    def get_word2idx(self):
        return self.word2idx

    def get_vocab(self):
        return self.vocab

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_label2idx(self):
        return self.label2idx

    def get_idx2label(self):
        return self.idx2label

    def get_output_size(self):
        return len(self.label2idx.keys())

    def get_corpus(self):
        return self.corpus

    def get_trainset(self):
        return self.train_dataset

    def get_testset(self):
        return self.test_dataset

    # signlon method
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(MovieReviewDataset, "_instance"):
            MovieReviewDataset._instance = MovieReviewDataset(*args, **kwargs)
        return MovieReviewDataset._instance


class MovieReviewDataLoader(Dataset):
    def __init__(self,is_train_set=True,occupy=0.7):
        self.is_train_set = True
        self.occupy = occupy
        self.mr_dataset = MovieReviewDataset.instance(self.is_train_set, self.occupy)
        if self.is_train_set:
            self.dataset = self.mr_dataset.get_trainset()
        else:
            self.dataset = self.mr_dataset.get_testset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]

    def get_vocab(self):
        return self.mr_dataset.get_vocab()

    def get_word2idx(self):
        return self.mr_dataset.get_word2idx()

    def get_label2idx(self):
        return self.mr_dataset.get_label2idx()

    def get_idx2word(self):
        return self.mr_dataset.get_idx2word()

    def get_idx2label(self):
        return self.mr_dataset.get_idx2label()

    def get_output_size(self):
        return len(self.get_label2idx().keys())

    def get_corpus(self):
        return self.mr_dataset.get_corpus()

    def get_max_seq_len(self):
        return self.mr_dataset.get_max_seq_len()
