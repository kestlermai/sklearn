# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:47:13 2022

@author: 11146
"""

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

#Day 1 数据预处理

#1.安装依赖环境
import numpy as np  #导入numpy包，简写为np，numpy包含数学计算函数
import pandas as pd  #导入pandas包，简写为pd，pandas包用于导入和管理数据集

#2.导入数据集，通过pandas导入数据集，注意路径
dataset = pd.read_csv('Day 1 Data.csv') #读取csv文件
X = dataset.iloc[ : , :-1].values  # 选择X数据集是第几行第几列 .iloc[行，列]
#iloc是一个函数，功能是根据标签的所在位置，python从左往右是从0开始计数，先选取行再选取列。
#从右往左是从1开始计数。
#这个[:,0:n]中的n是不算的，只读取到第n-1列
#dataset.iloc[1,1] = 取第二行第二列
#[ : ,0:n-1] = 第n-1列，n是不算的，所以[ : ,0:-1] = [ : ,0:3]，-1=3
Y = dataset.iloc[ : , 3].values  # : 表示全部行 or 列；[a]第a行 or 列
#.values 转化成python认的数值型，赋值X和Y
print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)
#print("X")这种带有双引号的，说明打印的是双引号包含的文字，属于字符型；

#3.处理丢失数据
#使用sklearn的内置方法Imputer，可以将丢失的数据用特定的方法补全。
from sklearn.impute import SimpleImputer#从sklearn包里面调用该函数
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')#缺失值用平均值补充
#当缺失值少于30%时，连续变量可以使用均值或中位数填补。分类变量也可以填补，单算一类即可，或者也可以用众数填补分类变量（就是哪一类多算哪一类）。
X[:,1:3] = imputer.fit_transform(X[:,1:3])#仅对第二列到第三列进行缺失值处理
X[:,1:-1] = imputer.fit_transform(X[:,1:-1])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

#4.解析分类结果
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#LabelEncoder标签编码，将字符型变量转化成数值型变量
#OneHotEncoder独热编码，将字符型变量去掉等级、大小，如红色为0,0,1 绿色为0,1,0 这样大家的距离都是根号2；类似加权
from sklearn.compose import ColumnTransformer
#ColumnTransformer数据转化，用于定义转化器的名称，
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])#对X第一列的France等字符进行转化
onehotencoder = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#根据第一列的分类进行独热编码
X = onehotencoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

#5.拆分数据集为训练集和测试集
#一般为80%训练集，20%测试集
#cross_validatio这个包早就不在使用了，划分到了model_selection这个包中。
#from sklearn.cross_validation import train_test_split 更改如下
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
#random state是设置随机种子，保证每次程序运行都是分割一样的训练集跟测试集

#6.特征缩放
#数据集中某些数据会很大，会导致数值越大，权重越大，但实际上无论数值多大，大家的权重应该都是一样的
#所以引入StandardScaler进行数据标准化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
