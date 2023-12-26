# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:58:51 2022

@author: 11146
"""

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径

#1.数据导入与处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Day 4 Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values#选取年龄跟薪资作为变量
Y = dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#将薪资跟年龄进行标准化，避免薪资数值过大影响模型，保持权重一致

#2.构建logistics模型
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
#logistics回归参数代码
#class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, 
                                              #fit_intercept=True, intercept_scaling=1, class_weight=None, 
                                              #random_state=None, solver='lbfgs', max_iter=100, 
                                              #multi_class='auto', verbose=0, warm_start=False, 
                                              #n_jobs=None, l1_ratio=None)
#penalty惩罚项,默认l2，解决过拟合
#dual对偶或原始方法，默认F，当样本数>样本特征时，dual默认为F。
#tol停止求解的标准，默认为1e-4
#C正则化强度的倒数，必须是一个大于0的浮点数，不填写默认1.0，即默认正则项与损失函数的比值是1：1。C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强，参数会逐渐被压缩得越来越小。
#fit_intercept，默认为T，是否计算模型的截距
#intercept_scaling=1，一个浮点数，只有当solver='liblinear'才有意义。当采用fit_intcept时相当于人造一个特征出来，特征恒为1，权重为b。在计算正则化项的时候，该人造特征也被考虑了，因此为了降低这个人造特征的影响，需要提供intercept_scaling。
#class_weight=None, 类别的权重，样本类别不平衡时使用，设置balanced会自动调整权重。为了平横样本类别比例，类别样本多的，权重低，类别样本少的，权重高。
#solver优化算法的参数，包括newton-cg,lbfgs,liblinear,sag,saga,对损失的优化的方法
#max_iter=100,最大迭代次数，
#multi_class=’ovr’,多分类方式，有‘ovr','mvm'
#verbose=0 输出日志，设置为1，会输出训练过程的一些结果
#warm_start=False热启动参数，如果设置为True,则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）
#l1_ratio=None，l1_ratio=0等价于使用penalty=‘l2’

#3.预测结果
Y_pred = classifier.predict(X_test)

#4.模型评估
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
#confusion_matrix混淆矩阵，是可视化工具，特别用于监督学习，无监督学习叫做匹配矩阵
#混淆矩阵的每一列代表了预测类别，每一列的总数表示预测为该类别的数据的数目；
#每一行代表了数据的真实归属类别，每一行的数据总数表示该类别的数据实例的数目；每一列中的数值表示真实数据被预测为该类的数目。