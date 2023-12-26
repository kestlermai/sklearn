# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 01:06:28 2023

@author: maihuanzhuo
"""

#误判病例分析
#在介绍AUC的时候，提到predict和predict_proba两个函数
#y_pred = classifier.predict(X_test)
#y_testprba = classifier.predict_proba(X_test)[:,1]
#y_test是实际值，y_testprba是预测的概率
#auc_test = roc_auc_score(y_test, y_testprba)
#所以呢，可以根据y_pred和y_true就可以判断是所谓的误诊（y_true是0，而y_pred是1）即假阳性，还是漏诊（y_true是1，而y_pred是0）即假阴性。
#以Xgboost为例子
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

import xgboost as xgb
param_grid=[{
            'n_estimators':[35],
            'eta':[0.1],
            'max_depth':[1],
            'gamma':[0],
            'min_child_weight':[5],
            'max_delta_step':[1],
            'subsample':[0.8],
            'colsample_bytree':[0.8],
            'colsample_bylevel':[0.8],
            'reg_lambda':[9],
            'reg_alpha':[5],
            },
           ]
boost = xgb.XGBClassifier()
classifier = xgb.XGBClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, n_jobs = -1, verbose = 2, cv=10)     
grid_search.fit(X_train, y_train)   
classifier = grid_search.best_estimator_ 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]

#由于我们演示的是测试集，所以关注y_pred和y_test:
#接着，我们需要再运行一次代码：
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 666)
#我们需要获得y_text对应的X_test，而之前运行的Xgboost代码的X_test的数据已经被归一化了，没法使用，我们需要的是原始数据，重新生成一次数据即可：

#https://mp.weixin.qq.com/s/waMjTZ2C4IVT8cU-lo-0nw 