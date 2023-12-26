# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:43:10 2023

@author: maihuanzhuo
"""

#LightGBM建模实战
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

#LightGBM的调参策略
# learning_rate：一般先设定为0.1，最后再作调整，合适候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]；
# max_depth：树的最大深度，默认值为-1，表示不做限制，合理的设置可以防止过拟合；
# num_leaves：叶子的个数，默认值为31，此参数的数值应该小于2^max_depth；
# min_data_in_leaf /min_child_samples：设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合，默认值20；
# min_split_gain：默认值为0，设置的值越大，模型就越保守，推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]；
# subsample：选择小于1的比例可以防止过拟合，但会增加样本拟合的偏差，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]；
# colsample_bytree：特征随机采样的比例，默认值为1，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]；
# reg_alpha：推荐的候选值为：[0, 0.01~0.1, 1]；
# reg_lambda：推荐的候选值为：[0, 0.1, 0.5, 1]；
# 参考大佬的调参策略：
# learning_rate设置为0.1；
# 调参：max_depth, num_leaves, min_data_in_leaf, min_split_gain, subsample, colsample_bytree；
# 调参：reg_lambda , reg_alpha；
# 降低学习率，继续调整参数，学习率合适候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]；

#先默认参数跑一遍

import lightgbm as lgb
classifier = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc')
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

# 明显过拟合
# 调整max_depth, num_leaves, min_data_in_leaf, min_split_gain, subsample, colsample_bytree：
# learning_rate设置为0.1，然后先来max_depth, num_leaves和min_data_in_leaf：

param_grid=[{
              'max_depth': [5, 10, 15, 20, 25, 30, 35],
              'num_leaves': range(5, 100, 5),
              'min_data_in_leaf': range(5,200,10),             
            },
           ]
boost = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc', learning_rate=0.1)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
#最优参数：max_depth=5, min_data_in_leaf=25, num_leaves=5
#再来min_split_gain, subsample和colsample_bytree
param_grid=[{
              'min_split_gain': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],             
            },
           ]

boost = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc', learning_rate=0.1, 
                           max_depth=5, min_data_in_leaf=25, num_leaves=5)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
#最优参数：colsample_bytree=0.7, min_split_gain=0.1,subsample=0.1

#确定lambda_l1和lambda_l2
param_grid=[{
              'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'reg_lambda': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],         
            },
           ]

boost = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc', learning_rate=0.1, 
                           max_depth=5,min_data_in_leaf=25, num_leaves=5, 
                           colsample_bytree=0.7, min_split_gain=0.1, subsample=0.1)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
#最优参数：reg_alpha=0.4, reg_lambda=0.3

#确定learning_rate
param_grid=[{
              'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3], 
            },
           ]

boost = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc', 
                           max_depth=5, min_data_in_leaf=25, num_leaves=5, 
                           colsample_bytree=0.7, min_split_gain=0.1, subsample=0.1,
                           reg_alpha=0.4, reg_lambda=0.3)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
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
#learning_rate 没变化，那就不调了

# eature_fraction、bagging_fraction、bagging_freq后续还可以调

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
plt.title('ROC of train')
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
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

