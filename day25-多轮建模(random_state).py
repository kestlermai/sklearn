# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:30:09 2023

@author: maihuanzhuo
"""

#多轮建模
#首先，之前提过，random_state这个参数，它的功能是确保每次随机抽样所得到的数据都是一样的，有利于数据的复现。
#比如，我们这十个ML模型，用的参数都是random_state=666，这样作比较才有可比性，因为训练集和验证集都是一样的，大家的起跑线一样，公平竞争。
#我之前也也给大家示范过，random_state选取不同的数值，模型的性能是有差别的，这也可以解释.
#毕竟我们演示的数据集样本量也就1000多，属于小样本，而且数据内部肯定存在异质性，因此，不同抽样的数据所得出来的模型性能，自然不同。
#所以，我觉得要综合判断一个模型好不好，一次随机抽样是不行的，得多次抽样建模，看看整体的性能如何才行（特别是对于这种小训练集）。
#因此我的思路是，随机抽取训练集和验证集2000次（随你），然后构建2000个ML模型（譬如2000个朴素贝叶斯），得出2000批性能参数。
#那怎么实现呢，还不就是random_state，下面上代码，以朴素贝叶斯为例：
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values
empty = np.array([[0],[0],[0],[0],[0],[0],[0]])#使用NumPy库创建7行1列的数组，每个元素为0
#7行分别是random_state抽样次数n，sen敏感度，sep特异度，auc，训练集的敏感度，特异度，auc
print(empty)
n=1
while n < 2001:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = n)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)   
    y_updatapred = classifier.predict(X_train)#训练集
    from sklearn.metrics import confusion_matrix, roc_auc_score
    cm = confusion_matrix(y_test, y_pred)   
    cm_updata = confusion_matrix(y_train, y_updatapred)#训练集混淆矩阵
    auc = roc_auc_score(y_test, y_pred)
    auc_updata = roc_auc_score(y_train, y_updatapred)#训练集auc
    a = cm[0,0]
    b = cm[0,1]
    c = cm[1,0]
    d = cm[1,1]
    sen = d/(d+c)#敏感度
    sep = a/(a+b)#特异度
    a_updata = cm_updata[0,0]
    b_updata = cm_updata[0,1]
    c_updata = cm_updata[1,0]
    d_updata = cm_updata[1,1]
    sen_updata = d_updata/(d_updata + c_updata)
    sep_updata = a_updata/(a_updata + b_updata)
    first = np.array([[n],[sen],[sep],[auc],[sen_updata],[sep_updata],[auc_updata]])#创建first数组
    second = np.hstack((empty,first))#np.hstack函数将empty和first数组水平堆叠在一起，即列与列连接
    empty = second
    n = n + 1
    print(n)
final_par = np.delete(second,0,axis=1)#删除第一列（Python中的索引从0开始计数），axis=1表示要沿着列的方向删除元素
print (final_par)
final_parT = final_par.T#行列置换=R中t()函数
#NumPy库没有np.savecsv函数
np.savetxt('NB_par.csv',final_parT,delimiter=',',#delimiter分割字符串，默认是任何空格，跟R中sep一样
           header='n,sen,sep,auc,sen_updata,sep_updata,auc_updata',
           fmt='%.3f')#fmt参数来指定格式字符串,'%d'表示设置为int型，'%s'表示为字符型，'%f'表示保留几位小数,"%.3f"保留三位小数
# '%e'浮点数格式，'%.18e',小数点后要显示的小数位数，即18位小数
#保存之后比如test-sen排个序，看看最好的有多好；比如看看2000次的平均值和标准差

