# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:57:08 2022

@author: 11146
"""
import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

#day2 线性回归

#1.导入数据
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Day 2 studentscores.csv')
X = dataset.iloc[:, : 1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=1/4, random_state=0)#测试集为25%
#种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

#2.训练模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#默认参数
regressor = regressor.fit(X_train, Y_train)
#线性回归代码
#class sklearn.linear_model.LinearRegression(*, fit_intercept=True, normalize=False, 
                                            #copy_X=True, n_jobs=None, positive=False)
#fit_intercept，默认为T，是否计算模型的截距
#normalize=标准化，默认F，若T则特征矩阵X在进入回归前将会被减去均值（中心化），并除以L2范式（缩放）
#若T，先用StandardScaler处理数据
#copy_X，默认T，将在X.copy上进行处理，否则特征矩阵X会被线性回归影响并覆盖
#n_jobs默认为none，用于计算的cpu，若为1则调用所有cpu进行计算
#positive默认F，若为T，将强制要求权重向量的分量为正数

#3.预测结果
Y_pred = regressor.predict(X_test)

#4.可视化
plt.scatter(X_train, Y_train, color='red', marker='*')
#scatter函数 散点图 散点图形marker='o'圈 or '.'点 or '*'星星
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

#测试集画图
plt.scatter(X_test,Y_test,color='red',marker='*')
plt.plot(X_test,Y_pred,color='orange')
plt.show()

#计算回归评价指标训练集跟测试集的MAE（平均绝对误差）、MAPE、MSE（均方根误差）、RMSE
from sklearn import metrics
import math

MAE = metrics.mean_absolute_error(Y_test, Y_pred)
print('MAE:',MAE)

MAPE = metrics.mean_absolute_percentage_error(Y_test, Y_pred)
print('MAPE:',MAPE)

MSE = metrics.mean_squared_error(Y_test,Y_pred)
print('MSE:',MSE)

RMSE = math.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:',RMSE)

R2 = metrics.r2_score(Y_test, Y_pred)
print('R2:',R2)

MAE1 = metrics.mean_absolute_error(X_train, Y_train)
print('MAE:',MAE1)
