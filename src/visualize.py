#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FileName: visualize.py
Description: visualization tools
Author: Barry Chow
Date: 2019/2/22 8:37 PM
Version: 0.1
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#from imblearn.over_sampling import BorderlineSMOTE
#from sklearn.model_selection import GridSearchCV
from config import result_filename
import numpy as np
from tools import load_attention_weights,load_contents
from visualization.tSNE_implementation import t_SNE_visualization
#global settings
myfont = fm.FontProperties(fname='/Users/Barry/Library/Fonts/SimHei.ttf')
epochs = 10
target_names = ['A','B','C']
#resampler = BorderlineSMOTE(random_state=10)
global_colors = ['darkgreen',"crimson",'dodgerblue','darkturquoise','dimgray','darkviolet','darkorange']

fig_path = '../figs/'
from tools import loadTextRepresentations,loadTextLabels
from visualization.attention_weights_visualization import showAttentionWeightsForContent

def visualization_acc(x_axis,data,filename,x_label='epochs'):
    plt.cla()
    plt.plot(x_axis,data,color='darkgreen',marker='o')
    #plt.plot(x_axis,data2,color="#ff6347",marker='o')
    #plt.ylim(0.6,0.9)
    plt.xlabel(x_label,fontproperties=myfont,fontsize='large')
    plt.ylabel("准确率",fontproperties=myfont,fontsize='large')
    plt.grid(ls='--')
    #plt.legend(["train acc","test acc"],loc='upper right',fontsize=10)

    #plt.show()
    plt.savefig(fig_path+filename)

#visualize attentive weights by using
def visualization_weights(text,weights):
    pass


#load all results
def loadAccResults():
    ret = []
    with open(result_filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            ret.append(line)
    return ret

'''
locate the accuracy results by six condition:
    'dataType','classifierType','learning rate','penalty confficient','isRand','isStatic'

'''
def searchDataByCondition(hyperparams,dataType,classifierType,type,range=40):
    data = loadAccResults()
    for line in data:
        if line[0]==type:
            if line[1]==dataType:
                if line[2]==classifierType:
                    if line[3]==str(hyperparams['isRand']):
                        if line[4]==str(hyperparams['isStatic']):
                            if line[5]==str(hyperparams['fine_tuned']):
                                if line[6]==str(hyperparams['LEARNING_RATE']):
                                    if line[7]==str(hyperparams['PENALTY_CONFFICIENT']):
                                        return [float(acc) for acc in line[8:8+range]]

#visualize two or more methods with same dataset
def visualizeMoreDataset(dataType, models, hyperparams,type='test_acc',xrange=40,ylabel="准确率"):
    filename = type+'_'+dataType + '_' + '_'.join(models) + "_" + str(hyperparams['LEARNING_RATE']).replace('.', '') +\
               '_' + str(hyperparams['PENALTY_CONFFICIENT']).replace('.', '') +'_' + str(hyperparams['isRand']) +'_' + \
               str(hyperparams['isStatic'])
    plt.cla()

    #get datas
    for idx in range(len(models)):
        model = models[idx]
        result = searchDataByCondition(hyperparams,dataType,model,type,xrange)
        #100 percent
        #result = [value*100 for value in result]
        x_axis = range(1, len(result) + 1)
        plt.plot(x_axis,result,color=global_colors[idx],marker='.')

    #plt.plot(x_axis, data1, color='darkgreen', marker='o')
    #plt.plot(x_axis,data2,color="#ff6347",marker='o')
    # plt.ylim(0.6,0.9)
    plt.xlabel('Epochs', fontproperties=myfont, fontsize='large')
    plt.ylabel(ylabel, fontproperties=myfont, fontsize='large')
    plt.grid(ls='--')
    plt.legend(models,loc='lower right',fontsize=10)

    # plt.show()
    plt.savefig(fig_path+ filename)


#visualize twin y axis in one plot
def visualizeTwinYAxis(dataType, model, hyperparams,xrange=40):
    filename = 'Twin_Y_'+dataType + '_' + model+ "_" + str(hyperparams['LEARNING_RATE']).replace('.', '') +\
               '_' + str(hyperparams['PENALTY_CONFFICIENT']).replace('.', '') +'_' + str(hyperparams['isRand']) +'_' + \
               str(hyperparams['isRand'])
    plt.cla()
    test_acc = searchDataByCondition(hyperparams, dataType, model, 'test_acc', range=100)
    train_acc = searchDataByCondition(hyperparams, dataType, model, 'train_acc', range=100)
    train_loss = searchDataByCondition(hyperparams, dataType, model, 'train_loss', range=100)
    x_axis = range(1,len(test_acc)+1)

    #visualize twin y axis
    fig = plt.figure()
    plt.xlabel('Epochs', fontproperties=myfont, fontsize='large')

    ax1 = fig.add_subplot(111)
    #ax1.plot(x_axis,test_acc,color=global_colors[0],marker='.', linestyle='-')
    #ax1.plot(x_axis,train_acc,color=global_colors[4],marker='.', linestyle='--')
    ax1.plot(x_axis, test_acc, color=global_colors[0],marker='.',lw=1.5)
    ax1.plot(x_axis, train_acc, color=global_colors[4],marker='.',lw=1.5)
    ax1.legend(['test accuracy','train accuracy'],loc='upper left',fontsize=10)
    ax1.set_ylabel('训练与测试准确率',fontproperties=myfont, fontsize='large')
    #ax1.set_title("Double Y axis")

    ax2 = ax1.twinx()  # this is the important function
    #ax2.plot(x_axis, train_loss, color=global_colors[6], marker='.', linestyle=':')
    ax2.plot(x_axis, train_loss, color='chocolate', marker='.',lw=1)
    ax2.legend(['train loss'], loc='upper right', fontsize=10)
    ax2.set_ylabel('单次训练损失', fontproperties=myfont, fontsize='large')

    plt.grid(ls='--')
    #plt.legend(['test accuracy'], loc='center right', fontsize=10)


    # plt.show()
    plt.savefig(fig_path + filename)


#visualize multi bar for model variations
def visualizeMultiBarForVariations():
    from src.config import hyperparams
    dataset = ['CR','MR','SST1','SST2', 'MPQA','TREC','Subj']
    #rand initialized
    hyperparams['isRand']=True
    hyperparams['isStatic']=True
    hyperparams['fine_tuned']=True
    rand_acc = locateHighestAcc(hyperparams,'FTIA',dataset,type='test_acc')
    print(dataset)

    print('FTIA-rand')
    print([round(value,4) for value in rand_acc])

    # static glove vector
    hyperparams['isRand'] = False
    hyperparams['isStatic'] = True
    hyperparams['fine_tuned'] = False
    static_acc = locateHighestAcc(hyperparams, 'FTIA',dataset, type='test_acc')
    print('FTIA-static')
    print([round(value,4) for value in static_acc])

    # non-static = static+fine_tuned
    hyperparams['isRand'] = False
    hyperparams['isStatic'] = True
    hyperparams['fine_tuned'] = True
    non_static_acc = locateHighestAcc(hyperparams, 'FTIA',dataset, type='test_acc')
    print('FTIA-non-static')
    print([round(value,4) for value in non_static_acc])

    x = np.arange(len(rand_acc))

    #visualize multi bar

    plt.cla()
    total_width, n = 0.8, 3
    width = total_width / n

    x = x - (total_width - width) / 2
    plt.bar(x, rand_acc, width=width, label='FTIA-rand',color='cornflowerblue',linewidth=0)
    #data label
    #for a, b in zip(x, rand_acc):
    #    plt.text(a+0.1, b , '%.2f' % round(100*b,2)+'%', ha='center', va='bottom', fontsize=5)

    plt.bar(x + width, static_acc, width=width, label='FTIA-static',color='orange',linewidth=0)
    # data label
    #for a, b in zip(x, static_acc):
    #    plt.text(a+width + 0.1, b, '%.2f' % round(100 * b, 2) + '%', ha='center', va='bottom', fontsize=5)

    plt.bar(x + 2 * width, non_static_acc, width=width, label='FTIA-non-static',color='darkgrey',linewidth=0)
    # data label
    #for a, b in zip(x, non_static_acc):
    #    plt.text(a +2*width+ 0.1, b, '%.2f' % round(100 * b, 2) + '%', ha='center', va='bottom', fontsize=5)

    plt.xticks(x+width*1.5,dataset,fontsize=14)

    plt.legend(loc='upper right', fontsize=11)
    plt.ylim(0,1.25)
    plt.ylabel('准确率',fontsize=14,fontproperties=myfont)

    #plt.show()
    plt.savefig(fig_path+'FTIA_variations')


def locateHighestAcc(hyperparams,classifierType,dataset,type='test_acc'):
    ret_result = []
    for data in dataset:
        result = searchDataByCondition(hyperparams, data, classifierType, type, range=100)
        ret_result.append(max(result))
    return np.array(ret_result)

#report highest accuracy in each dataset
def reportHighestAcc(hyperparams):
    #dataset = ['CR','MR','SST1','SST2', 'MPQA','TREC','Subj']
    #classifierset = ['TextCNN','LSTM','BiGRU', 'LSTMAtt', 'RCNN', 'SelfAtt', 'FTIA']
    #dataset = ['UsptoPatent']
    #classifierset = ['FTIA','BiGRU','LSTMAtt','SelfAtt','TextCNN']

    dataset = ['CR', 'SST1', 'Subj', 'TREC', 'UsptoPatent']
    classifierset = ['LSTM','FTIA']
    for data in dataset:
        for classifier in classifierset:
            result = searchDataByCondition(hyperparams,data,classifier,type='test_acc',range=100)
            print(data,' --- ',classifier,' --- ',round(max(result),4)*100,'%')
        print('')

#tSNE visualization input data
def tSNE_visualize(datasetType,classifierType,epoch=1):
    textRepr = loadTextRepresentations(datasetType,classifierType)
    textLabels = loadTextLabels(datasetType,classifierType)
    t_SNE_visualization(datasetType,classifierType,textRepr,
                        textLabels,{0:'a',1:'b',2:'c',3:'d',4:'e',5:'f'},epoch)

#visualize attention weights
def visualize_Attention(datasetType,classifierType,penalty):
    reviews = load_contents(datasetType,classifierType,penalty)
    weights = load_attention_weights(datasetType,classifierType,penalty)
    start= 21

    #length = 2


    #positive
    #review_idx = [103,104,7003,2010,2011]

    #negative
    #review_idx = [2103,2021,2133,2064,2074]

    review_idx = range(start,start+5)

    visualize_reviews = []
    visualize_weights = []
    for idx in review_idx:
        visualize_reviews.append(reviews[idx])
        visualize_weights.append(weights[idx])

    showAttentionWeightsForContent(visualize_reviews,visualize_weights,penalty)

    #showAttentionWeightsForContent(reviews[start:start+length],weights[start:start+length],penalty)


def visualize_compare_Attention(datasetType, classifierType1, classifierType2):
    type1_reviews = load_contents(datasetType, classifierType1)
    type1_weights = load_attention_weights(datasetType, classifierType1)

    type2_reviews = load_contents(datasetType, classifierType2)
    type2_weights = load_attention_weights(datasetType, classifierType2)

    #start=2010
    #review_idx = range(start, start + 5)


    # positive
    #review_idx = [103,104,7003,2010,2011]
    #review_idx = [104,7003,2010,2011]

    # negative
    #review_idx = [2103,2021,2133,2064,2074]
    #review_idx = [2103,2021,2133,2074]

    #TREC
    #review_idx = [1009,2012,1508,5010]

    # CR
    review_idx = [59,225,147,155]

    #start=1508
    #review_idx =range(start,start+5)


    type1_visualize_reviews = []
    type1_visualize_weights = []
    for idx in review_idx:
        type1_visualize_reviews.append(type1_reviews[idx])
        type1_visualize_weights.append(type1_weights[idx])

    showAttentionWeightsForContent(type1_visualize_reviews, type1_visualize_weights,classifierType1)

    #locate the corresponding review,weights in type2
    type2_visualize_reviews = []
    type2_visualize_weights = []
    for review in type1_visualize_reviews:
        correspond_idx = find_review_idx(review,type2_reviews)
        print(correspond_idx)
        if correspond_idx==-1:
            print('###Error not found review',review)
        type2_visualize_reviews.append(type2_reviews[correspond_idx])
        type2_visualize_weights.append(type2_weights[correspond_idx])
    showAttentionWeightsForContent(type2_visualize_reviews, type2_visualize_weights,classifierType2)

def compare_list(list1,list2):
    if len(list1)==len(list2):
        for i in range(len(list1)):
            if list1[i]==list2[i]:
                pass
            else:
                return False
    else:
        return False
    return True

def find_review_idx(single_review,reviews):
    for idx,review in enumerate(reviews):
        if compare_list(single_review,review):
            return idx
    return -1


#visualize different penalty for model FTIA in all 7 dataset
def visualize_different_penalty():
    from .config import hyperparams
    dataset = ['CR', 'MR', 'SST1', 'SST2', 'MPQA', 'TREC', 'Subj']
    #dataset  = ['CR','MR']
    print('penalty: ')
    print([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.cla()
    for dataset_idx in range(len(dataset)):
        datasetType = dataset[dataset_idx]
        high_acc_with_penalty = []
        #penalty in [0,0.1,0.2,...,1.0]
        for penalty in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            hyperparams['PENALTY_CONFFICIENT']=penalty
            acc_result = searchDataByCondition(hyperparams,datasetType,'FTIA',type='test_acc',range=100)
            high_acc_with_penalty.append(max(acc_result))

        print('dataset: ',datasetType)
        print([round(value,4) for value in high_acc_with_penalty])
        #x_axis = range(1, len(high_acc_with_penalty) + 1)
        x_axis = np.arange(0, 1.1, 0.1)
        plt.plot(x_axis, high_acc_with_penalty, color=global_colors[dataset_idx], marker='.')

    plt.xlabel('Penalty Confficient', fontproperties=myfont, fontsize='large')
    plt.ylabel("准确率", fontproperties=myfont, fontsize='large')
    plt.grid(ls='--',axis ='y')
    plt.legend(dataset, loc='center right', fontsize=10)
    #plt.ylim(0.4,1.05)
    plt.xlim(-0.05,1.05)

    # plt.show()
    plt.savefig(fig_path + 'FTIA_different_penalty')

#visualize runing times for one epoch
def visualize_running_time():
    #value = [2.2,0.5,1.5,4.1,2.3,5.3,3.9]
    #classifierSet = ['FTIA','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt']

    value = [2.2, 0.5, 4.1, 2.3, 3.9]
    plt.cla()
    '''
    #N = 20
    N = len(run_times)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    #radii = 10 * np.random.rand(N)
    #width = np.pi / 4 * np.random.rand(N)
    width = [0.5]*N
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(1)
    bars = ax.bar(theta, run_times, width=width, bottom=0.0)
    # 前三个参数分别对应left,height,width,left表示从哪开始，表示开始位置，height表示从中心点向边缘绘制的长度
    for r, bar in zip(run_times, bars):
        bar.set_facecolor(plt.cm.viridis(r / 2.))
        bar.set_alpha(0.5)
    plt.show()
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #name =['FTIA-static','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt']  # 标签
    name =['FTIA','TextCNN','BiGRU','LSTMAtt','SelfAtt']  # 标签

    theta = np.linspace(0, 2 * np.pi, len(name), endpoint=False)  # 将圆根据标签的个数等比分
    theta = np.concatenate((theta, [theta[0]]))  # 闭合
    value = np.concatenate((value, [value[0]]))  # 闭合

    ax = plt.subplot(111, projection='polar')  # 构建图例
    ax.plot(theta, value, 'darkgreen', lw=2, ls='--',alpha=0.75,marker='o')  # 绘图
    #ax.fill(theta, value, 'm', alpha=0.75)  # 填充
    ax.set_thetagrids(theta * 180 / np.pi, name)  # 替换标签
    ax.set_ylim(0, 6)  # 设置极轴的区间
    ax.set_theta_zero_location('N')  # 设置极轴方向
    #ax.set_title('CR数据集下各模型运行时间对比图', fontsize=16)  # 添加图描述
    #plt.show()
    plt.savefig(fig_path+'Models_runtime_comp')

def visualize_params_num():
    #plt.cla()
    plt.figure(figsize=(8, 6))
    #plt.ylim(0, 5)
    #plt.xlim(0, 10)
    x_change = 0.5
    y_change = 0.25
    xticks_size = 11
    plt.xticks(np.arange(0, 10 ,2), ['0%', "5%", "10%", "15%", "20%", "25%"])

    name_list = ['FTIA','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt']
    num_list = np.array([4349402,3676232,4308802,4870802,4308802,5392202,16392082])/2500000
    print(num_list)
    plt.barh(range(len(num_list)), num_list, color='darkgreen', tick_label=name_list, align='center')
    for a, b in zip(range(len(num_list)), num_list):
        plt.text(b + x_change, a - y_change, '%.2f' % round(b /1000000, 2), ha='center', va='bottom',
                 fontsize=xticks_size)
    plt.show()
    #plt.savefig("../figs/" + "Visualize_parmams_nums")

#visualize case study for different labels using radia
def visualize_case_stduy(isCorrect = True):
    plt.cla()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    name = ['A', 'B', 'C', 'F', 'H', 'G']  # 标签
    theta = np.linspace(0, 2 * np.pi, len(name), endpoint=False)  # 将圆根据标签的个数等比分
    theta = np.concatenate((theta, [theta[0]]))  # 闭合
    #lstm
    lstm_correct_value = [769,695,735,808,690,741]  # 在60-120内，随机取5个数
    lstm_error_value = [82,227,187,95,217,154]
    #ftia
    ftia_correct_value = [811,648,737,798,761,788]  # 在
    ftia_error_value = [98,285,178,71,189,126]

    if isCorrect:
        lstm_values = lstm_correct_value
        ftia_values = ftia_correct_value
        type = "correct"
    else:
        lstm_values = lstm_error_value
        ftia_values = ftia_error_value
        type = 'error'

    value = np.concatenate((lstm_values, [lstm_values[0]]))
    ax = plt.subplot(111, projection='polar')  # 构建图例
    ax.plot(theta, value, 'darkorange', lw=2, ls='-', alpha=0.75, marker='o')  # 绘图
    #ftia
    value = np.concatenate((ftia_values, [ftia_values[0]]))
    ax = plt.subplot(111, projection='polar')  # 构建图例
    ax.plot(theta, value, 'darkgreen', lw=2, ls='-', alpha=0.75, marker='o')  # 绘图

    # ax.fill(theta, value, 'm', alpha=0.75)  # 填充
    ax.set_thetagrids(theta * 180 / np.pi, name,fontsize=24)  # 替换标签
    #ax.set_ylim(0, 60)  # 设置极轴的区间
    ax.set_theta_zero_location('N')  # 设置极轴方向

    #ax.spines['polar'].set_visible(False)  # 将轴隐藏
    #ax.grid(axis='y')  # 只有y轴设置grid

    # 设置X轴的grid
    '''n_grids = np.linspace(0, 1, 6, endpoint=True)  # grid的网格数
    grids = [[i] * (len(lstm_correct_value)) for i in n_grids]  # grids的半径

    for i, grid in enumerate(grids[:-1]):  # 给grid 填充间隔色
        ax.plot(name, grid, color='grey', linewidth=0.5)
        if (i > 0) & (i % 2 == 0):
            ax.fill_between(name, grids[i], grids[i - 1], color='grey', alpha=0.1)
    '''
    plt.legend(['LSTM','FTIA'],loc='upper right', fontsize=10)
    # ax.set_title('CR数据集下各模型运行时间对比图', fontsize=16)  # 添加图描述
    #plt.show()
    plt.savefig(fig_path + type+'_case_study_comp.png')



if __name__ =='__main__':
    from config import hyperparams

    reportHighestAcc(hyperparams)

    #src.main import hyperparams
    #visualizeMoreDataset('MPQA',['LSTM','LSTMAtt','BiGRU','RCNN','SelfAtt','FTIA'],hyperparams,type='test_acc',range=40,ylabel="准确率")
    #visualizeMoreDataset('SST1',['FTIA','LSTM'],hyperparams)

    '''for  data in ['Subj','TREC','SST1','SST2','MPQA','CR','MR']:
    #for data in ['TREC']:
        #visualizeMoreDataset(data,['FTIA','LSTMAtt'],hyperparams,type='test_acc',xrange=100,ylabel='测试准确率')

        visualizeMoreDataset(data,['FTIA','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt'],hyperparams,type='test_acc',xrange=100,ylabel='测试准确率')
        visualizeMoreDataset(data,['FTIA','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt'],hyperparams,type='train_loss',xrange=100,ylabel='训练损失')
        visualizeMoreDataset(data,['FTIA','TextCNN','LSTM','BiGRU','LSTMAtt','RCNN','SelfAtt'],hyperparams,type='train_acc',xrange=100,ylabel='训练准确率')
    '''
    #report accuracy
    #reportHighestAcc(hyperparams)

    #visualize train loss and train accuracy
    #for data in ['Subj','TREC','SST1','SST2','MPQA','CR','MR']:
    #    visualizeTwinYAxis(data,'FTIA',hyperparams,100)

    #visualize FTIA model variations
    #visualizeMultiBarForVariations()

    #visualize tSNE for text repesentations
    #tSNE_visualize('TREC','FTIA')

    #visualize different penalty
    #visualize_different_penalty()

    #visualize_running_time()

    #visualize_Attention('MR','FTIA',0.1 )
    #visualize_Attention('MR', 'FTIA', 0.1)
    #visualize_Attention('MR','SelfAtt',0.1)
    #visualize_compare_Attention('MR','FTIA','SelfAtt',0.1)
    #visualize_compare_Attention('TREC', 'FTIA', 'SelfAtt')

    #visualize_compare_Attention('CR', 'FTIA', 'SelfAtt')



    #visualize params nums
    #visualize_params_num()

    #hyperparams['LEARNING_RATE']=1e-3
    #visualizeMoreDataset('UsptoPatent', ['FTIA','LSTM','LSTMAtt'], hyperparams,type='test_acc', xrange=50, ylabel='训练准确率')

    #reportHighestAcc(hyperparams)

    #case study
    #visualize_case_stduy(True)
    #visualize_case_stduy(False)