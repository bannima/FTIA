#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: ModelFactory.py
Description: Model factory
Author: Barry Chow
Date: 2019/3/18 4:51 PM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import abc
import logging
from models.BiGRU import BiGRU
from models.FTIA import FTIA
from models.TextCNN import TextCNNClassifer
from models.LSTM import LSTMClassifier
from models.LSTM_Attention import LSTM_Attention
from models.RCNN import RCNNClassifier
from models.SelfAttention import SelfAttentionClassifier

class AbstractModelFactory(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_model_classifier(self,kwargs):
        pass

class LSTMFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return LSTMClassifier(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'],word_vector=kwargs['word_vector'])

class LSTMAttFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return LSTM_Attention(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'])

class BiGRUFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return BiGRU(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'])

class TextCNNFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return TextCNNClassifer(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'])


class RCNNFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return RCNNClassifier(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'])


class SelfAttFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return SelfAttentionClassifier(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'],datasetType=kwargs['datasetType'])


class FTIAFactory(AbstractModelFactory):
    def get_model_classifier(self,kwargs):
        return FTIA(input_size=kwargs['input_size'], hidden_size=kwargs['HIDDEN_SIZE'], output_size=kwargs['output_size'],
                    n_layers=kwargs['N_LAYERS'], word_vector=kwargs['word_vector'], fine_tuned=kwargs['fine_tuned'],datasetType=kwargs['datasetType'])

def loadClassifierByType(classifierType,kwargs):
    if classifierType=='LSTM':
        return LSTMFactory().get_model_classifier(kwargs)
    elif classifierType =='LSTMAtt':
        return LSTMAttFactory().get_model_classifier(kwargs)
    elif classifierType=='BiGRU':
        return BiGRUFactory().get_model_classifier(kwargs)
    elif classifierType == 'TextCNN':
        return TextCNNFactory().get_model_classifier(kwargs)
    elif classifierType=='RCNN':
        return RCNNFactory().get_model_classifier(kwargs)
    elif classifierType=='SelfAtt':
        return SelfAttFactory().get_model_classifier(kwargs)
    elif classifierType=='FTIA':
        return FTIAFactory().get_model_classifier(kwargs)
    else:
        logging.error("Not recognized classifier type",classifierType)
