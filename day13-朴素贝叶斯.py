# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:06:20 2022

@author: 11146
"""

#朴素贝叶斯（Naive Bayes，NB），属于分类算法，有监督但不建模
#算法得出的结论，永远不是100%确定的，更多的是判断出了一种“样本的标签更可能是某类的可能性”，而非一种“确定”
#高斯朴素贝叶斯GaussianNB即正态分布

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

#导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 导入数据集
dataset = pd.read_csv('Day 4 Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values
# 性别转化为数字
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# 将数据集分成训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#构建NB模型
from sklearn.naive_bayes import GaussianNB#导入NB算法
classifier = GaussianNB(var_smoothing=1e-9)
classifier.fit(X_train,Y_train)
#class sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)
#priors：先验概率大小，如果没有给定，模型则根据样本数据自己计算（利用极大似然法）。
#var_smoothing：所有特征的最大方差部分，添加到方差中用于提高计算稳定性，默认1e-9。

#预测结果
Y_pred = classifier.predict(X_test)

#评估模型
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
