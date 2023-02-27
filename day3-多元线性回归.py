# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:25:16 2022

@author: 11146
"""

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

#1.导入数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Day 3 50_Startups.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder = LabelEncoder()
X[:,3]=LabelEncoder.fit_transform(X[:,3])
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#2.构建线性模型
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#3.预测结果
y_pred = regressor.predict(X_test)
print('y_real:',Y_test)
print('y_pred:',y_pred)
map_index = list(range(len(Y_test)))
#range函数返回一系列连续添加的整数，能够生成一个列表对象
#len函数用于返回字符串、列表、字典、元组等长度
plt.scatter(map_index, Y_test, color='red',marker='*')
plt.plot(map_index, y_pred, color='orange')
plt.show()