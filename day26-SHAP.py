# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 00:32:38 2023

@author: maihuanzhuo
"""
#https://zhuanlan.zhihu.com/p/85791430

#SHAP
#来自chatGPT的解释：
#SHAP（SHapley Additive exPlanations）是一种用于解释机器学习模型预测的开源库和方法。
#它旨在帮助理解模型的预测结果，揭示每个特征对于模型预测的贡献程度。
#SHAP基于合作博弈理论中的Shapley值概念，将其应用于特征的重要性评估和模型解释。
#SHAP的主要功能和应用包括：
#1.特征重要性：SHAP可以计算每个特征对于模型的预测结果的重要性。这有助于识别哪些特征对于模型的决策具有更大的影响力。
#2.预测解释：SHAP可以解释单个样本的预测结果，说明为什么模型对于特定输入产生了特定的输出。这有助于理解模型的决策过程。
#3.模型解释：SHAP可以提供整体模型的解释，帮助分析模型的整体行为和特征之间的相互作用。
#4.图形可视化：SHAP提供了丰富的可视化工具，使解释结果更容易理解和传达。
import shap#使用pip install shap进行安装

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
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#绘画SHAP相关图
import shap
explainer = shap.TreeExplainer(classifier)
shap.initjs()
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)#SHAP摘要图，结合了特征重要度和特征的影响
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.force_plot(explainer.expected_value, shap_values, X_train)#??
#可以和Xgboost自带的重要指数相比较

##单样本解析???后续再看解释
shap.force_plot(explainer.expected_value, shap_values[2,:], X_train[2,:], #iloc 是Pandas数据框的索引方法，用于基于行和列的位置来选择数据
                X_train[0], matplotlib=True)
