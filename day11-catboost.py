# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:52:34 2022

@author: 11146
"""

#CatBoost，全称大概是这个Gradient Boosting（梯度提升） + Categorical Features（类别型特征）
#它对分类型特征有自己独到的处理方法，省得我们我们在筛选变量的时候纠结于连续变量和分类变量的相关性
#预测偏移处理，从而减少模型的过拟合

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径

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

#构建catboost
import catboost as cb
#具体可以看：https://catboost.ai/en/docs/references/training-parameters/
#loss_function：指损失函数，默认Logloss。可选项：RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。
#custom_loss：指定训练过程中计算显示的损失函数。可选项：Logloss、CrossEntropy、Precision、Recall、F、F1、BalancedAccuracy、AUC等。
#eval_metric：指定用于过度拟合检测和最佳模型选择的指标。可选项同custom_loss。
#iterations：迭代次数，默认500。
#learning_rate：学习速度，默认0.03。
#l2_leaf_reg：L2正则化。
#bootstrap_type：确定抽样时的样本权重，默认GPU下为Bayesian、CPU下为MVS。可选项：Bayesian、Bernoulli(伯努利实验)、MVS(仅支持cpu)、Poisson(仅支持gpu)、No（取值为No时，每棵树为简单随机抽样）。
#bagging_temperature：bootstrap_type=Bayesian时使用，取值为1时采样权重服从指数分布；取值为0时所有采样权重均等于1。取值范围[0，inf)，值越大、bagging就越激进。
#subsample：样本采样比率（行采样）。
#sampling_frequency：采样频率，仅支持CPU。可选：PerTree（在构建每棵新树之前采样）、PerTreeLevel（默认值，在子树的每次分裂之前采样）。
#random_strength：特征分裂信息增益的扰动项，默认1，用于避免过拟合。
#use_best_model：使用效果最优的子树棵树/迭代次数，使用验证集的最优效果对应的迭代次数（eval_metric：评估指标，eval_set：验证集数据）。
#best_model_min_trees：最少子树棵树。和use_best_model一起使用。
#depth：树深度，默认6。
#grow_policy：子树生长策略。可选：SymmetricTree（默认值，对称树）、Depthwise（整层生长，同xgb）、Lossguide（叶子结点生长，同lgb）。
#min_data_in_leaf：叶子结点最小样本量。
#max_leaves：最大叶子结点数量。
#rsm：列采样比率，默认值1，取值（0，1]。
#nan_mode：缺失值处理方法。可选：Forbidden（不支持缺失值，输入包含缺失时会报错）、Min（处理为该列的最小值，比原最小值更小）、Max（处理为该列的最大值，比原最大值更大）。
#input_borders：特征数据边界（最大最小边界），会影响缺失值的处理（nan_mode取值Min、Max时），默认值None。
#class_weights：y标签类别权重。用于类别不均衡处理，默认均为1。
#auto_class_weights：自动计算平衡各类别权重。
#scale_pos_weight：二分类中第1类的权重，默认值1（不可与class_weights、auto_class_weights同时设置）。
#boosting_type：特征排序提升类型。可选项：Ordered（catboost特有的排序提升，在小数据集上效果可能更好，但是运行速度较慢）、Plain（经典提升）
#feature_weights：特征权重，在子树分裂时计算各特征的信息增益该特征权重。设置方式：1）feature_weights = [0.1, 1, 3]；2）feature_weights = {"Feature2":1.1,"Feature4":0.3}。

#category参数
#max_ctr_complexity：指定分类特征交叉的最高阶数，默认值4。

#ouput参数
#logging_level：选择输出什么信息，可选项：Silent（不输出信息）、Verbose（默认值，输出评估指标、已训练时间、剩余时间等）、Info（输出额外信息、树的棵树）、Debug（debug信息）。
#metric_period：指定计算目标值、评估指标的频率，默认值1。
#verbose：输出日记信息，类似于logging_level（两者只设置一个），取值True对应上方Verbose、False对应Silent。

#过拟合检测参数
#early_stopping_rounds：早停设置，默认不启用。
#od_type：过拟合检测类型，默认IncToDec。可选：IncToDec、Iter。
#od_pval：IncToDec过拟合检测的阈值，当达到指定值时，训练将停止。要求输入验证数据集，建议取值范围[10e-10，10e-2s]。默认值0，即不使用过拟合检测。

#数值型变量分箱设置参数
#border_count：分箱数，默认254。
#feature_border_type：分箱方法，默认GreedyLogSum。可选：Median、Uniform、UniformAndQuantiles、MaxLogSum、MinEntropy、GreedyLogSum。

#不加Categorical Features
Clf1 = cb.CatBoostClassifier(eval_metric='AUC', depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
Clf1.fit(X_train, Y_train)
#加Categorical Features（不需要再进行标化缩放）
Clf2 = cb.CatBoostClassifier(eval_metric='AUC', depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
cat_features_index = [0] #性别在第0列
Clf2.fit(X_train, Y_train, cat_features=cat_features_index )
#告诉模型分类变量有多少种类，分别用什么数值表示。比如SUV这个数据集，性别是分类变量，有“男”、“女”两类，用数值0和1代替，在第0列，所以代码为：cat_features_index = [0]。
#结果预测
Y_pred1 = Clf1.predict(X_test)
Y_pred2 = Clf2.predict(X_test)

#模型评估
cm1 = confusion_matrix(Y_test, Y_pred1)
print(cm1)

cm2 = confusion_matrix(Y_test, Y_pred2)
print(cm2)
