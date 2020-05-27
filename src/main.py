#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: main.py
Description:  main project
Author: Barry Chow
Date: 2019/1/29 10:44 AM
Version: 0.1
"""
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from DatasetFactory import loadDatsetByType
from ModelFactory import loadClassifierByType
from tools import create_variable,loadWordVectors,saveCSVResult
from tools import saveTextRepresentations,saveTextLabels
from tools import delMiddleReprLabelFiles,delMiddleContentWeightFiles
from tools import save_attention_weights,save_contents
import csv
from config import result_filename
from config import hyperparams
from visualization.tSNE_implementation import t_SNE_visualization
import numpy as np
from tools import loadTextRepresentations,loadTextLabels
from config import global_datasetType,global_classifierType
from datetime import datetime

def str2arr(content,word2idx):
    content = [word2idx[word] for word in content.split()]
    return content,len(content)

#pad sequence and sort the tensor
def pad_sequences(vectorized_seqs,seq_lengths,labels,global_same_length):
    #CNN requires same seq length
    if global_same_length:
        seq_tensor = torch.zeros((len(vectorized_seqs),hyperparams['global_max_seq_len'])).long()
    else:
        seq_tensor = torch.zeros((len(vectorized_seqs),seq_lengths.max())).long()
    for idx,(seq,seq_len) in enumerate(zip(vectorized_seqs,seq_lengths)):
        seq_tensor[idx,:seq_len] = torch.LongTensor(seq)

    #sort tensors by their length
    seq_lengths,perm_idx = seq_lengths.sort(0,descending=True)
    seq_tensor = seq_tensor[perm_idx]

    #also sort label in the same order
    labels = torch.LongTensor(labels)[perm_idx]

    #return variables
    #data,parallel requires everything to be a variable
    return create_variable(seq_tensor),create_variable(seq_lengths),create_variable(labels)

#create necessary variables
def make_variables(reviews,labels,word2idx,label2idx,global_same_length):
    #numerical sequences
    sequence_and_length = [str2arr(review,word2idx) for review in reviews]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])

    #numerical labels
    labels = [label2idx[label] for label in labels]
    return pad_sequences(vectorized_seqs,seq_lengths,labels,global_same_length)

def train(train_loader,word2idx,label2idx,penalty_confficient,global_same_length,classifier,criterion,optimizer,datasetType,classifierType):
    total_loss = 0
    correct=0

    train_data_size = len(train_loader.dataset)
    for i,(reviews,labels) in enumerate(train_loader,1):
        #save reviews
        #save_contents(datasetType, classifierType, reviews)

        input,seq_lengths,labels = make_variables(reviews,labels,word2idx,label2idx,global_same_length)

        #save text labels
        saveTextLabels(datasetType,classifierType,labels.data.cpu().numpy())

        #output = classifier(input,seq_lengths,labels)
        output,penalty = classifier(input,seq_lengths,labels)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss = criterion(output,labels)+penalty_confficient*penalty
        #loss = criterion(output,labels)
        total_loss += loss.item()

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

        if i%500 ==0:
            print(datetime.now().strftime( '%Y-%m-%d %H:%M:%S')," ### i:",i,"loss: ",loss.item(),', penalty item: ',penalty_confficient*penalty)
            #print(datetime.datetime.now().strftime( '%Y-%m-%d %H:%M:%S')," ### i:",i,"loss: ",loss.item())
    print("*** train  accuracy: ", float(correct) / train_data_size, ' train loss: ',total_loss)

    return float(correct) / train_data_size, total_loss

def test(test_loader,word2idx,label2idx,global_same_length,classifier,idx2label):
    #global global_datasetType, globale_classifierType
    #calc weights for words
    '''if review:
        input,seq_lenghts,label = make_variables([review],[label],word2idx,label2idx)
        output= classifier(input,seq_lenghts,label)
        #output,penalty = classifier(input,seq_lenghts,label)
        pred = output.data.max(1, keepdim=True)[1]
        print('label: ',label,'pred: ',pred)

        #print(review)
        #print(weights)
        #print()
        return
    '''
    print("--- evaluating trained model ---")
    correct=0
    test_data_size = len(test_loader.dataset)
    #count correct preds for each label
    correct_count = {}
    #error count
    error_count = {}
    for reviews,labels in test_loader:
        input,seq_lenghts,labels = make_variables(reviews,labels,word2idx,label2idx,global_same_length)

        # save text labels
        #saveTextLabels(global_datasetType, globale_classifierType, labels.data.numpy())
        #save reviews
        #save_contents(global_datasetType,globale_classifierType,reviews,0.1)

        #FTIA model
        #output = classifier(input,seq_lenghts,labels)
        output,penalty = classifier(input,seq_lenghts,labels)
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        #calc correct ones and error ones
        for idx,res in enumerate(pred.eq(labels.data.view_as(pred)).cpu().numpy()):
            true_label = str(int(labels[idx]))
            #pred correct
            if res[0]==1:
                if true_label in correct_count:
                    correct_count[true_label]+=1
                else:
                    correct_count[true_label]=1
            #false
            elif res[0]==0:
                if true_label in error_count:
                    error_count[true_label]+=1
                else:
                    error_count[true_label]=1
            else:
                raise ValueError("not recognized result")

    print("-------- correct count---------")
    for true_label,true_count in sorted(correct_count.items(),key=lambda asv:asv[1]):
        print(idx2label[int(true_label)],' --- ',true_count)
    print("-------- error count---------")
    for true_label,error_count in sorted(error_count.items(),key=lambda asv:asv[1]):
        print(idx2label[int(true_label)],' --- ',error_count)

    print("*** test accuracy: ",float(correct)/test_data_size)
    return float(correct)/test_data_size

#run model
def run_model_with_hyperparams(hyperparams,datasetType,classifierType,imp_tSNE=True):

    #set global dataset and classifiertype for tSNE

    hyperparams['datasetType']=datasetType
    hyperparams['classifierType']=classifierType

    print("*** dataset= ",datasetType,' classifier= ',classifierType,' ***')
    # hyper params
    HIDDEN_SIZE = hyperparams['HIDDEN_SIZE']
    N_LAYERS = hyperparams['N_LAYERS']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    N_EPOCHS = hyperparams['N_EPOCHS']
    LEARNING_RATE = hyperparams['LEARNING_RATE']
    # weight to adjust penalty loss
    PENALTY_CONFFICIENT = hyperparams['PENALTY_CONFFICIENT']

    #only text CNN requires same length
    global_same_length = classifierType=='TextCNN'

    # laod dataset by type "TREC","Subj","CR","MR","SST1","SST2","MPQA"
    train_dataset, test_dataset = loadDatsetByType(datasetType)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    vocab = train_dataset.get_vocab()
    word2idx = train_dataset.get_word2idx()
    label2idx = train_dataset.get_label2idx()
    idx2label = train_dataset.get_idx2label()
    input_size = len(vocab.keys())
    output_size = train_dataset.get_output_size()

    #set hyperparams
    hyperparams['input_size']=input_size
    hyperparams['output_size']=output_size
    hyperparams['word_vector'] = loadWordVectors(word2idx,datasetType,HIDDEN_SIZE,isStatic=hyperparams['isStatic'],isRand=hyperparams['isRand'])
    hyperparams['global_max_seq_len']=train_dataset.get_max_seq_len()

    # load classifier by classifier type, LSTM, LSTMAtt,BiGRU,TextCNN,RCNN,SelfAtt,FTIA
    classifier = loadClassifierByType(classifierType, hyperparams)
    #move model to GPU
    if torch.cuda.is_available():
        classifier=classifier.cuda()

    # report model parmams num
    calc_model_parmams_nums(classifierType, classifier)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(classifier.parameters(),lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=LEARNING_RATE)


    #run model and test
    test_acc_with_epochs = []
    train_acc_with_epochs = []
    train_loss_with_epochs = []
    for epoch in range(1,N_EPOCHS+1):
        #del middle files
        delMiddleReprLabelFiles('TREC', classifierType)
        delMiddleContentWeightFiles(datasetType,classifierType)
        starttime = datetime.now()
        print('/n')
        print("### epoch: ",epoch," ",datetime.now().strftime( '%Y-%m-%d %H:%M:%S'))

        train_acc,train_loss = train(train_loader,word2idx,label2idx,PENALTY_CONFFICIENT,global_same_length,classifier,criterion,optimizer,datasetType,classifierType)
        train_acc_with_epochs.append(train_acc)
        train_loss_with_epochs.append(train_loss)
        test_acc_with_epochs.append(test(test_loader,word2idx,label2idx,global_same_length,classifier,idx2label))
        endtime = datetime.now()
        print(global_datasetType, ' --- one batch needs seconds ---', (endtime - starttime).seconds)

        #which epochs to visualize using tSNE

        if imp_tSNE and epoch in [1,5,10,20,30,40,70,100]:
            textRepr = loadTextRepresentations('TREC',classifierType)
            textLabels = loadTextLabels('TREC',classifierType)
            t_SNE_visualization('TREC',classifierType,textRepr,textLabels,idx2label,epoch)


    #save train ,test accuracy and train loss result
    saveCSVResult(hyperparams,datasetType,classifierType,test_acc_with_epochs,type='test_acc')
    saveCSVResult(hyperparams,datasetType,classifierType,train_acc_with_epochs,type='train_acc')
    saveCSVResult(hyperparams,datasetType,classifierType,train_loss_with_epochs,type='train_loss')

def calc_model_parmams_nums(modelName,model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        #print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        #print("该层参数和：" + str(l))
        k = k + l
    print(modelName," 总参数数量和：" + str(k))


if __name__ =='__main__':

    dataset=['CR','MR','SST1','SST2','MPQA','TREC','Subj']
    classifierset = ['LSTM','LSTMAtt','BiGRU','RCNN','SelfAtt','FTIA']

    '''for data in dataset:
        for classifer in classiferset:
            for isRand in [True,False]:
                hyperparams['isRand']=isRand
                run_model_with_hyperparams(hyperparams,datasetType=data,classifierType=classifer)
    '''
    #run_model_with_hyperparams(hyperparams,datasetType='CR',classifierType='LSTM')

    #GPU computer

    '''for classifier in ['LSTM','LSTMAtt','BiGRU','RCNN','FTIA','SelfAtt']:
    #for classifier in ['FTIA','SelfAtt']:
        #for data in ['CR','MR','Subj','MPQA']:
        #for data in ['SST2','SST1','TREC']:
        for data in ['AGNews','TwentyNews']:
           run_model_with_hyperparams(hyperparams,datasetType=data,classifierType=classifier)
    '''
    #local computer
    #for classifier in ['LSTM','LSTMAtt','BiGRU','RCNN','FTIA','SelfAtt']:
    '''for classifier in ['FTIA','SelfAtt']:
        #for data in ['CR','MR','Subj','MPQA']:
        for data in ['SST2','SST1','TREC']:
        #for data in ['AGNews','TwentyNews']:
           run_model_with_hyperparams(hyperparams,datasetType=data,classifierType=classifier)
    '''
    #all dataset ['Subj','TREC','SST1','SST2','MPQA','CR','MR','AGNews','TwentyNews']
    '''for data in ['MR']:
        run_model_with_hyperparams(hyperparams,datasetType=data,classifierType='TextCNN')
    '''

    '''
    #FTIA model variations
    for data in ['CR,'MR','SST1','SST2','MPQA','Subj','TREC']:
        # FTIA-rand
        hyperparams['isRand']=True
        hyperparams['isStatic']=True
        hyperparams['fine_tuned']=True
        run_model_with_hyperparams(hyperparams,datasetType=data,classifierType='FTIA')

        #FTIA-non-static
        hyperparams['isRand']=False
        hyperparams['isStatic']=True
        hyperparams['fine_tuned']=True
        run_model_with_hyperparams(hyperparams, datasetType=data, classifierType='FTIA')
    
    '''

    '''
    #different penalty
    for data in ['CR','MR','SST1','SST2','MPQA','Subj','TREC']:
        for penalty in [0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            hyperparams['PENALTY_CONFFICIENT']=penalty
            run_model_with_hyperparams(hyperparams, datasetType=data, classifierType='FTIA')
    '''

    '''
    #tSNE visualization are limited on FTIA and LSTMAtt
    run_model_with_hyperparams(hyperparams,datasetType='TREC',classifierType='FTIA',imp_tSNE=True)
    run_model_with_hyperparams(hyperparams,datasetType='TREC',classifierType='LSTMAtt',imp_tSNE=True)
    '''


    #count batch times
    #p in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #hyperparams['PENALTY_CONFFICIENT']=0.1
    #run_model_with_hyperparams(hyperparams,datasetType='Subj',classifierType='LSTM')

    #test the uspto patent dataset
    #FTIA-static
    '''
    run_model_with_hyperparams(hyperparams,datasetType='UsptoPatent',classifierType='FTIA')

    # FTIA-rand
    hyperparams['isRand'] = True
    hyperparams['isStatic'] = True
    hyperparams['fine_tuned'] = True
    run_model_with_hyperparams(hyperparams, datasetType='UsptoPatent', classifierType='FTIA')

    # FTIA-non-static
    hyperparams['isRand'] = False
    hyperparams['isStatic'] = True
    hyperparams['fine_tuned'] = True
    run_model_with_hyperparams(hyperparams, datasetType='UsptoPatent', classifierType='FTIA')
    '''

    #patent classification task
    #run_model_with_hyperparams(hyperparams, datasetType='UsptoPatent', classifierType='LSTM')
    #run_model_with_hyperparams(hyperparams, datasetType='UsptoPatent', classifierType='LSTMAtt')
    #run_model_with_hyperparams(hyperparams, datasetType='UsptoPatent', classifierType='FTIA')

    #attention weight visualization
    #for model in ['SelfAtt','FTIA','TextCNN','BiGRU','LSTMAtt']:
    #    run_model_with_hyperparams(hyperparams,datasetType='UsptoPatent',classifierType=model)


    #attention weight for TREC
    #run_model_with_hyperparams(hyperparams, datasetType='TREC', classifierType='SelfAtt')

    # attention weight for CR
    #run_model_with_hyperparams(hyperparams, datasetType='CR', classifierType='FTIA')

    #add BiLSTM
    #for data in ['CR','SST1','Subj','TREC','UsptoPatent']:
    #    for classifier in ['LSTM','FTIA']:
    #       run_model_with_hyperparams(hyperparams,datasetType=data,classifierType=classifier)

    # add scatter figure for SelfAtt and FTIA using TREC dataset
    for data in ['TREC']:
        for classifier in ['SelfAtt','FTIA','LSTMAtt']:
            run_model_with_hyperparams(hyperparams,datasetType=data,classifierType=classifier)