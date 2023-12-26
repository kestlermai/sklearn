# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:08:41 2022

@author: 11146
"""

#Xgboost建模

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

#xgboost调参
#包括eta、gamma、max_depth、min_child_weight、max_delta_step、subsample
#colsample_bytree、colsample_bylevel、lambda、alpha、n_estimators。
#n_estimators：基础模型数量，都懂了；
#eta：类似于Adaboost的leanring_rate，eta通过缩减特征的权重使提升计算过程更加保守（越不容易过拟合），默认0.3；
#max_depth：树的最大深度，值越大，越容易过拟合，默认6；
#gamma：值越大，算法越保守（越不容易过拟合），默认0，范围(0,1]；
#min_child_weight：值较大时，可以避免模型学习到局部的特殊样本，这个值过高，会导致欠拟合，默认1；
#max_delta_step：设置正值算法会更保守（越不容易过拟合），默认0；
#subsample：减小这个值算法会更加保守，避免过拟合，但是设置的过小，它可能会导致欠拟合，默认1，范围(0,1]；
#colsample_bytree：每颗树随机采样的列数的占比，默认1，范围(0,1]；
#colsample_bylevel：对列数的采样的占比，默认1，范围(0,1]；
#lambda：L2 正则化项的权重系数，越大模型越保守，默认1；
#alpha：L1 正则化项的权重系数，越大模型越保守，默认0；

#先默认xgboost跑一遍
import xgboost as xgb
classifier = xgb.XGBClassifier()
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
#出现了过拟合，调参

#换一种调法，一个一个参数的调整，主要是解决过拟合问题了：
#先调n_estimators
import xgboost as xgb
param_grid=[{
            'n_estimators':[i for i in range(100,1000,100)],
            },
           ]
boost = xgb.XGBClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#n_estimators=100，需要往小调
import xgboost as xgb
param_grid=[{
            'n_estimators':[i for i in range(10,150,5)],
            },
           ]
boost = xgb.XGBClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#n_estimators=15
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#到eta，类似于Adaboost的leanring_rate，eta通过缩减特征的权重使提升计算过程更加保守（越不容易过拟合），默认0.3；

param_grid=[{
            'eta':[0.01,0.02,0.04,0.08,0.1,0.2,0.3],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#eta=0.3
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#似乎好像没变化

#到max_depth
param_grid=[{
            'max_depth':[i for i in range(1,10,1)],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15,eta=0.3)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#max_depth=2
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#好像不错

#接着到gamma，gamma：值越大，算法越保守（越不容易过拟合），默认0，范围(0,1]；
param_grid=[{
            'gamma':[0,0.1,1.0,2.0,3.0],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#gamma=0

#min_child_weight，值较大时，可以避免模型学习到局部的特殊样本，这个值过高，会导致欠拟合，默认1；
param_grid=[{
            'min_child_weight':[i for i in range(1,10,1)],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#min_child_weight=4
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#没啥变化

#max_delta_step默认0，这个参数限制了每棵树权重改变的最大步长，
#如果这个参数的值为0,则意味着没有约束。如果他被赋予了某一个正值，则是这个算法更加保守。
#通常，这个参数我们不需要设置，但是当个类别的样本极不平衡的时候，这个参数对逻辑回归优化器是很有帮助的。
#感觉这个可以跳过
param_grid=[{
            'max_delta_step':[i for i in range(1,10,1)],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#max_delta_step=2
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#subsample减小这个值算法会更加保守，避免过拟合，但是设置的过小，它可能会导致欠拟合，默认1，范围(0,1]；
param_grid=[{
            'subsample':[0.1,0.2,0.4,0.6,0.8,1.0],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          max_delta_step=2)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
print(grid_search.best_params_)
#subasample=0.8
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#微小变化

#colsample_bytree每颗树随机采样的列数的占比，默认1，范围(0,1]；
param_grid=[{
            'colsample_bytree':[0.1,0.2,0.4,0.6,0.8,1.0],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          max_delta_step=2,subasample=0.8)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#colsample_bytree=0.8
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#又变回去了

#colsample_bylevel对列数的采样的占比，默认1，范围(0,1]；
param_grid=[{
            'colsample_bylevel':[0.1,0.2,0.4,0.6,0.8,1.0],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          max_delta_step=2,subasample=0.8,colsample_bytree=0.8)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
#默认参数1,那没必要看结果了

#reg_lambda也称reg_lambda,默认值为0，权重的L2正则化项。(和Ridge regression类似)。
#这个参数是用来控制XGBoost的正则化部分的。这个参数在减少过拟合上很有帮助。
param_grid=[{
            'reg_lambda':[i for i in range(2,50,2)],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          max_delta_step=2,subasample=0.8,colsample_bytree=0.8)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
print(grid_search.best_params_)
#reg_lambda': 5
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#没变化

#reg_alpha也称reg_alpha默认为0，权重的L1正则化项。(和Lasso regression类似)。 
#可以应用在很高维度的情况下，使得算法的速度更快。
param_grid=[{
            'reg_alpha':[i for i in range(1,50,2)],
            },
           ]
boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          max_delta_step=2,subasample=0.8,colsample_bytree=0.8,reg_lambda=5)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
print(grid_search.best_params_)
#reg_alpha': 3
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#变化也不大
#综上最优参数为boost = xgb.XGBClassifier(n_estimators=15, eta=0.3, max_depth=2,gamma=0,min_child_weight=4,
                          #max_delta_step=2,subasample=0.8,colsample_bytree=0.8,reg_lambda=5,reg_alpha=3)

#综合调整一遍                          
import xgboost as xgb
param_grid=[{
            'n_estimators':[15,25],
            'eta':[0.2,0.4],
            'max_depth':[1,2],
            'gamma':[0,0.1],
            'min_child_weight':[4,6],
            'max_delta_step':[1,2],
            'subsample':[0.8,1.0],
            'colsample_bytree':[0.8,1.0],
            'reg_lambda':[5,7],
            'reg_alpha':[3,5],
            },
           ]
boost = xgb.XGBClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
print(grid_search.best_params_)
#'colsample_bytree': 0.8, 'eta': 0.4, 'gamma': 0, 'max_delta_step': 1, 'max_depth': 2,
# 'min_child_weight': 4, 'n_estimators': 15, 'reg_alpha': 3, 'reg_lambda': 5, 'subsample': 0.8
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
