#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: DatasetFactory.py
Description: dataset loader using factory pattern
Author: Barry Chow
Date: 2019/3/18 4:04 PM
Version: 0.1
"""

import abc
from .dataloader import MovieReviewDataLoader
from .dataloader import TRECDataLoader
from .dataloader import TwentyNewsDataLoader
from .dataloader import SubjDatasetLoader
from .dataloader import MPQADataLoader
from .dataloader import AGNewsDataLoader
from .dataloader import CustomerReviewDataLoader
from .dataloader import SST1DataLoader
from .dataloader import SST2DataLoader
from .dataloader import UsptoPatentDataLoader
import logging


class AbstractDatasetFactory(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_train_test_dataset(self):
        pass

class MovieReviewDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return MovieReviewDataLoader(is_train_set=True), MovieReviewDataLoader(is_train_set=False)

class TRECDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return TRECDataLoader(is_train_set=True),TRECDataLoader(is_train_set=False)

class SubjDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return SubjDatasetLoader(is_train_set=True),SubjDatasetLoader(is_train_set=False)

class MPQADatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return MPQADataLoader(is_train_set=True), MPQADataLoader(is_train_set=False)

class CustomerReviewDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return CustomerReviewDataLoader(is_train_set=True),CustomerReviewDataLoader(is_train_set=False)

class SST1DatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return SST1DataLoader(is_train_set=True), SST1DataLoader(is_train_set=False)

class SST2DatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return SST2DataLoader(is_train_set=True), SST2DataLoader(is_train_set=False)

class AGNewsDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return AGNewsDataLoader(is_train_set=True),AGNewsDataLoader(is_train_set=False)

class TwentyNewsDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return TwentyNewsDataLoader(is_train_set=True),TwentyNewsDataLoader(is_train_set=False)

class UsptoPatentDatasetFactory(AbstractDatasetFactory):
    def get_train_test_dataset(self):
        return UsptoPatentDataLoader(is_train_set=True),UsptoPatentDataLoader(is_train_set=False)

#load dataset by type
def loadDatsetByType(type):
    if type=='TREC':
        return TRECDatasetFactory().get_train_test_dataset()
    elif type=='Subj':
        return SubjDatasetFactory().get_train_test_dataset()
    elif type=='MPQA':
        return MPQADatasetFactory().get_train_test_dataset()
    elif type=='CR':
        return CustomerReviewDatasetFactory().get_train_test_dataset()
    elif type=='MR':
        return MovieReviewDatasetFactory().get_train_test_dataset()
    elif type=='SST1':
        return SST1DatasetFactory().get_train_test_dataset()
    elif type=='SST2':
        return SST2DatasetFactory().get_train_test_dataset()
    elif type=='AGNews':
        return AGNewsDatasetFactory().get_train_test_dataset()
    elif type=='TwentyNews':
        return TwentyNewsDatasetFactory().get_train_test_dataset()
    elif type=='UsptoPatent':
        return UsptoPatentDatasetFactory().get_train_test_dataset()
    else:
        logging.error("Not recognized dataset type "+type)

