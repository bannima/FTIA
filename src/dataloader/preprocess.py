#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: preprocess.py
Description: preprocess dataset
Author: Barry Chow
Date: 2019/1/29 10:52 AM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re


def clean_str(line):
    string = re.sub(r"[^A-Za-z0-9(),\.!?]", " ", line)
    # string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"e\.g\.,", " ", string)
    string = re.sub(r"a\.k\.a\.", " ", string)
    string = re.sub(r"i\.e\.,", " ", string)
    string = re.sub(r"i\.e\.", " ", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"\'", "", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"u\.s\.", " us ", string)
    return string.strip().lower()
