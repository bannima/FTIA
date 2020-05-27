#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: attention_weights_visualization.py
Description: 
Author: Barry Chow
Date: 2019/3/26 4:08 PM
Version: 0.1
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#from imblearn.over_sampling import BorderlineSMOTE
#from sklearn.model_selection import GridSearchCV
#from config import result_filename

import numpy as np
#global settings
import math
myfont = fm.FontProperties(fname='/Users/Barry/Library/Fonts/SimHei.ttf')
epochs = 10
#resampler = BorderlineSMOTE(random_state=10)
global_colors = ['darkgreen',"crimson",'dodgerblue','darkturquoise','dimgray','darkviolet','darkorange']

'''
fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.text(6, 10, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 10})

ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

ax.text(3, 2, 'unicode: Institut für Festkörperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

plt.show()
'''

'''
#show word with weights in anywhere and any color
def showSingleWordwithWeight((x,y),word,weights,plt):
    plt.text(x, y, word, size=10, rotation=0,
             ha="right", va="top",
             bbox=dict(boxstyle="square",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       alpha = 0.1,
                       )
             )
'''

#show batch contents and weights
def showAttentionWeightsForContent(contents,attweights,classifierType):
    #transform numpy to list

    plt.cla()
    plt.rcParams['figure.figsize'] = (16.0, 4.0)
    #fig = plt.figure(figsize=(8, 6))
    fig = plt.figure()
    fig.suptitle(' Visualization of Attention Weights', fontsize=14, fontweight='bold')

    #ax = fig.add_subplot(111)
    unit = 0.0082
    x,y = 0.05,0.90
    for content,attweight in zip(contents,attweights):
        #drop the useless weigths
        attweight = attweight[:len(content)]
        #min max scaler
        attweight = [(weight-min(attweight))/(max(attweight)-min(attweight)) for weight in attweight]
        count =0
        for word,weight in zip(content,attweight):
            plt.text(x, y, word, style='italic',fontsize=12,
                    bbox={'facecolor': 'red', 'alpha': weight, 'pad': 2})
            x+=len(word)*unit+0.01
            count +=1
            #every 10 words,change the line
            if count%14==0:
                x = 0.1
                y -=0.09

            #print x,len(word),'---',word,' --- ',weight
            print(word)
        print(' ')
        #reset coordinates
        x = 0.05
        y -= 0.14
    #plt.show()
    plt.savefig('../figs/'+'comp_positive_'+classifierType+'_weights_visualization')


if __name__ =='__main__':
    contents = [['I','Love','Apple','.'],['I',"don't",'Love','Apple','.']]
    attweights = [[0.1,0.4,0.5,0.05],[0.1,1.0,0.4,0.5,0.05]]
    showAttentionWeightsForContent(contents,attweights,0.1)
