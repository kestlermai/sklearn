# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:11:02 2023

@author: maihuanzhuo
"""

#朴素贝叶斯建模实战
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 666)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#NB的调参策略
# priors：先验概率大小，如果没有给定，模型则根据样本数据自己计算（利用极大似然法），这个可以不调。
# var_smoothing：所有特征的最大方差部分，添加到方差中用于提高计算稳定性，默认1e-9。

#先默认参数跑一遍：
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

import math
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)   
cm_train = confusion_matrix(y_train, y_trainpred)
#测试集的参数
a = cm[0,0]
b = cm[0,1]
c = cm[1,0]
d = cm[1,1]
acc = (a+d)/(a+b+c+d)
error_rate = 1 - acc
sen = d/(d+c)
sep = a/(a+b)
precision = d/(b+d)
F1 = (2*precision*sen)/(precision+sen)
MCC = (d*a-b*c) / (math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))
auc_test = roc_auc_score(y_test, y_testprba)
#训练集的参数
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train)))
auc_train = roc_auc_score(y_train, y_trainprba)  
#auc_test=0.8424894673173987, auc_train=0.8746981697485469, b=40
#结果还行

#调参var_smoothing：
param_grid=[{
             'var_smoothing': [1e-9,1e-6,1e-4,1e-3,1e-2,1,10,100],
           },
           ]
boost = GaussianNB()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#auc_test=0.8479122345972552, auc_train=0.8815802927458491, b=32
grid_search.best_estimator_
#GaussianNB(var_smoothing=1)

import math
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)   
cm_train = confusion_matrix(y_train, y_trainpred)
#测试集的参数
a = cm[0,0]
b = cm[0,1]
c = cm[1,0]
d = cm[1,1]
acc = (a+d)/(a+b+c+d)
error_rate = 1 - acc
sen = d/(d+c)
sep = a/(a+b)
precision = d/(b+d)
F1 = (2*precision*sen)/(precision+sen)
MCC = (d*a-b*c) / (math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))
auc_test = roc_auc_score(y_test, y_testprba)
#训练集的参数
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train)))
auc_train = roc_auc_score(y_train, y_trainprba)  

#绘画训练集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_trainprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_train), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('ROC Curve of Train')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#绘画测试集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_test, y_testprba, pos_label=1, drop_intermediate=False)  
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_test), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')   
plt.title('ROC Curve of Test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#绘画测试集混淆矩阵
classes = list(set(y_test))
classes.sort()
plt.imshow(cm_test, cmap=plt.cm.Blues)
indices = range(len(cm_test))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Prediction')
plt.ylabel('Truth')
for first_index in range(len(cm_test)):
    for second_index in range(len(cm_test[first_index])):
        plt.text(first_index, second_index, cm_test[first_index][second_index])
 
plt.show() 

#绘画训练集混淆矩阵
classes = list(set(y_train))
classes.sort()
plt.imshow(cm_train, cmap=plt.cm.Blues)
indices = range(len(cm_train))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Prediction')
plt.ylabel('Truth')
for first_index in range(len(cm_train)):
    for second_index in range(len(cm_train[first_index])):
        plt.text(first_index, second_index, cm_train[first_index][second_index])
 
plt.show()

#绘画测试集PR曲线
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
precision_1, recall_1, thresholds = precision_recall_curve(y_test, y_testprba)
plt.step(recall_1, precision_1, color='darkorange', alpha=0.2,where='post')
plt.fill_between(recall_1, precision_1, step='post', alpha=0.2,color='darkorange')
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_1,precision_1)
plt.show()
AP_test = average_precision_score(y_test, y_testprba, average='macro', pos_label=1, sample_weight=None)

#绘画训练集PR曲线
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
precision_1, recall_1, thresholds = precision_recall_curve(y_train, y_trainprba)
plt.step(recall_1, precision_1, color='darkorange', alpha=0.2,where='post')
plt.fill_between(recall_1, precision_1, step='post', alpha=0.2,color='darkorange')
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_1,precision_1)
plt.show()
AP_train = average_precision_score(y_train, y_trainprba, average='macro', pos_label=1, sample_weight=None)
