#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: patent_preprocess.py
Description: 
Author: Barry Chow
Date: 2019/4/4 10:58 AM
Version: 0.1
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json

classdef = {
    'A':3000,
    'B':3000,
    'C':3000,
    'G':3000,
    'F':3000,
    'H':3000
}

if __name__ =='__main__':
    path = '../dataset/UsptoPatent/'
    filename = 'uspto_patent.json'
    error_count = 0
    class_count = {
        'A':0,
        'B':0,
        'C':0,
        'G':0,
        'F':0,
        'H':0
    }
    for line in open(path+filename):
        try:
            patent = json.loads(eval(line).replace('\'', '\"'))
            classtype = patent['patent_classify']
            if classtype in classdef:
                if class_count[classtype]<classdef[classtype]:
                    title = patent['patent_title']
                    abstract = patent['patent_abstract']
                    #save patent
                    with open(path+'patent_sample2.txt','a+') as f:
                        f.write(' '.join([classtype,title,abstract])+'\n')
                        #print 'save patent'
                    class_count[classtype]+=1


            '''if classtype in class_count:
                class_count[classtype]+=1
            else:
                class_count[classtype]=1
            '''
            #print patent['patent_number']
        except Exception as e:
            error_count+=1
    print 'error object ',error_count
    print class_count
