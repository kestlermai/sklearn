# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:13:47 2022

@author: 11146
"""

#输入数据有标签，则为有监督学习；没标签则为无监督学习。
#监督学习是指数据集的正确输出已知情况下的一类学习算法。因为输入和输出已知，意味着输入和输出之间有一个关系，监督学习算法就是要发现和总结这种“关系”。
#无监督学习是指对无标签数据的一类学习算法。因为没有标签信息，意味着需要从数据集中发现和总结模式或者结构。
#有监督学习（分类，回归）-半监督学习（分类，回归）-半监督聚类（有标签数据的标签不是确定的，类似于：肯定不是xxx，很可能是yyy）-无监督学习（聚类）
#有监督机器学习的核心是分类，无监督机器学习的核心是聚类（将数据集合分成由类似的对象组成的多个类）。
#有监督的工作是选择分类器和确定权值，无监督的工作是密度估计（寻找描述数据统计值），这意味着无监督算法只要知道如何计算相似度就可以开始工作。
#独立分布数据更适合有监督，非独立数据更适合无监督。

#支持向量机（Support Vector Machine, SVM），属于有监督的机器学习，最主要用于分类问题中，属于二分类算法
#原理：根据特征值，构建一个n维空间（n个特征值），把每个数据点投影到该空间内
#通过算法计算出一个最佳超平面（泛化能力最好、鲁棒性最强即稳定性），用于数据分类

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径

#导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC#SVC算法
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

#模型训练
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier = SVC(kernel = 'poly', random_state = 0)
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier = SVC(kernel = 'precomputed', random_state = 0)#precomputed需要方阵计算
classifier.fit(X_train, Y_train)
#class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
                    #probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
                    #max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
#c：浮点数，默认为1.0。正则化参数。正则化的强度与C成反比。必须严格为正。此惩罚系数是l2惩罚系数的平方。
#kernel：{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}，默认为’rbf’。
#使用的核函数，必须是“linear”，“poly”，“rbf”，“sigmoid”，“precomputed”或者“callable”中的一个。

#degree：整数，默认3。多项式核函数的次数(' poly ')。将会被其他内核忽略。
# gamma：浮点数或者{‘scale’, ‘auto’}，默认为’scale’。核系数包含‘rbf’,‘poly’和‘sigmoid’。
#如果gamma='scale'(默认)，则它使用1 / (n_features * X.var())作为gamma的值，如果是auto，则使用1 / n_features。

#coef0：浮点数，默认为0.0。核函数中的独立项。它只在' poly '和' sigmoid '中有意义。
#shrinking：布尔值，默认为True。是否使用缩小启发式。
#probability：布尔值，默认为False。是否启用概率估计。必须在调用fit之前启用此参数，因为该方法内部使用5折交叉验证，
#因此会减慢该方法的速度，并且predict_proba可能与dict不一致。

#tol：浮点数，默认1e-3。残差收敛条件。
#cache_size：浮点数，默认200。指定内核缓存的大小（以MB为单位）。
#class_weight：{dict, ‘balanced’}, 默认None。在SVC中，将类i的参数C设置为class_weight [i] * C。
#如果没有给出值，则所有类都将设置为单位权重。“balanced”模式使用y的值自动将权重与类频率成反比地调整为n_samples / (n_classes * np.bincount(y))。

#verbose：布尔值，默认False。是否启用详细输出。
#max_iter：整数型，默认-1。对求解器内的迭代进行硬性限制，或者为-1（无限制时）。
#decision_function_shape：{‘ovo’, ‘ovr’}, 默认’ovr’。
#是否要将返回形状为(n_samples, n_classes)的one-vs-rest (‘ovr’)决策函数应用于其他所有分类器，
#而在多类别划分中始终使用one-vs-one (‘ovo’)，对于二进制分类，将忽略该参数。

#break_ties：布尔值，默认False。如果为true，decision_function_shape ='ovr'，并且类数> 2，则预测将根据Decision_function的置信度值打破平局；
#否则，返回绑定类中的第一类。请注意，与简单的预测相比，打破平局的计算成本较高。

#random_state：整数型，默认None。控制用于数据抽取时的伪随机数生成。


#预测结果
Y_pred = classifier.predict(X_test)

#评估模型
cm = confusion_matrix(Y_test, Y_pred)
print(cm)