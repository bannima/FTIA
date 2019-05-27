#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: config.py
Description: global settings
Author: Barry Chow
Date: 2019/1/29 10:46 AM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np

#global  settings
'''hyperparams = {
    'HIDDEN_SIZE': 200,
    'BATCH_SIZE': 32,
    'N_LAYERS': 2,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 2e-5,
    'PENALTY_CONFFICIENT': 0.1,
    'isRand': False,
    'isStatic': True,
    'fine_tuned':False,
}'''

hyperparams = {
    'HIDDEN_SIZE': 200,
    'BATCH_SIZE': 32,
    'N_LAYERS': 2,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 1e-3,
    'PENALTY_CONFFICIENT': 0.1,
    'isRand': False,
    'isStatic': True,
    'fine_tuned':False,
}

global_datasetType = ''
global_classifierType = ''

result_filename = './all_results.csv'
#result_filename = './all_results_patent.csv'
#result_filename = './all_results_penalty.csv'
#result_filename = './all_results_variations.csv'


