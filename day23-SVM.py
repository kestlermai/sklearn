# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:41:04 2023

@author: maihuanzhuo
"""

#支持向量机（SVM）建模实战
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

#SVM比较简单
# kernel：{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}，默认为’rbf’。
#使用的核函数，必须是“linear”，“poly”，“rbf”，“sigmoid”，“precomputed”或者“callable”中的一个。
# c：浮点数，默认为1.0。正则化参数。正则化的强度与C成反比。必须严格为正。一般可以选择为：10^t , t=[- 4，4]就是0.0001 到10000。
# gamma：核系数为‘rbf’,‘poly’和‘sigmoid’才可以设置。可选择下面几个数的倒数：0.1、0.2、0.4、0.6、0.8、1.6、3.2、6.4、12.8。
# degree：整数，默认3。多项式核函数的次数(' poly ')。将会被其他内核忽略。

#先默认参数跑一遍
from sklearn.svm import SVC
classifier = SVC(random_state = 0, probability=True)
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

#居然不错，没有过拟合情况
#auc_train=0.931, auc_test=0.852, b=34(一类错误)

#kernel='rbf'，只需要调gamma：
param_grid=[{
            'gamma':[10, 5, 2.5, 1.5, 1.25, 0.625, 0.3125, 0.15, 0.05, 0.025, 0.0125],             
            },
           ]

boost = SVC(kernel='rbf', random_state = 0, probability=True)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#gamma=0.05
#auc_test=0.859高了一点点

#kernel='linear'，没参数：
classifier = SVC(kernel='linear', random_state = 0, probability=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#差了点，不过感觉不错auc_test=0.8527092979602052

#kernel='sigmoid'，调整参数C（大写的C）和gamma：
param_grid=[{
            'gamma':[1,2,3,4,5,6,7,8,9,10],  
            'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000], 
            },
           ]
boost = SVC(kernel='sigmoid', random_state = 0, probability=True)
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#C=0.01, gamma=1
#auc_test=0.845784841279773，结果变差了

#kernel='poly'，调整参数C（大写的C）、gamma和d：
classifier = SVC(kernel='poly', random_state = 0, probability=True)
classifier.get_params().keys()
#dict_keys(['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 
#'degree','gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])

#先调整参数C（大写的C）、gamma：
param_grid=[{
            'gamma':[1,2,3,4,5,6,7,8,9,10],  
            'C':[0.0001,0.001,0.01,0.1,1,10,100,1000,10000], 
            },
           ]
boost = SVC(kernel='poly', random_state = 0, probability=True)
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#C=0.001, gamma=1
#auc_test=0.8264297334501314, auc_train=0.9491548752959312

#再调整degree：
param_grid=[{
            "degree":[1,2,3,4,5,6,7,8,9,10], 
            },
           ]
boost = SVC(C=0.001, gamma=1, kernel='poly', probability=True, random_state=0)
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 1, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#degree还是等于3

##综合来看，可能还是rbf核还有liner核的性能好一些。SVM还是厉害哦，至少在这个数据集，调参容易快捷，性能也到位。
