# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 14:35:40 2022

@author: 11146
"""

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

#KNN（K-Nearest Neihbor）算法，K-近邻算法,
#通过K值=主观设定，而且应该取奇数，K值不能取太小，比如K=1的情况，此时容易被错误的样本干扰，造成过拟合。
#当然，K值也不能取太大，越大分类效果越差，造成欠拟合。
#通过K值确定距离进行分类

#1.入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier#KNN分类算法
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#2.数据导入、处理
dataset = pd.read_csv('Day 4 Social_Network_Ads.csv') 
X=dataset.iloc[:,1:4].values
Y=dataset.iloc[:,4].values
LabelEncoder=LabelEncoder()
X[:,0]=LabelEncoder.fit_transform(X[:,0])#将性别转化为数值

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()#标化
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#3.构建KNN模型
classifier = KNeighborsClassifier(n_neighbors=5,)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
#KNeighborsClassifier(n_neighbors=5,weights=’uniform’,algorithm=’auto’,leaf_size=30,p=2,
                        #metric=’minkowski’,metric_params=None,n_jobs=1,**kwargs)
#n_neighbors=5,可选参数（默认为5）用于kneighbors查询的默认邻居的数量；
# weights（权重）,默认为uniform统一权重,distance’: 权重点等于他们距离的倒数。使用此函数，更近的邻居对于所预测的点的影响更大。[callable]: 一个用户自定义的方法，此方法接收一个距离的数组，然后返回一个相同形状并且包含权重的数组
#alogorithm算法，{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选参数（默认为‘auto’）。‘ball_tree’是为了克服kd树高纬失效而发明的，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。‘kd_tree’构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一个超矩形，在维数小于20时效率高。‘brute’使用暴力搜索.也就是线性扫描，当训练集很大时，计算非常耗时。‘auto’会基于传入fit方法的内容，选择最合适的算法。
# leaf_size（叶子数量）：int，可选参数（默认为30）。
#p：integer,默认为2。p = 1, 相当于使用曼哈顿距离（l1），p = 2，相当于使用欧几里得距离（l2）对于任何 p ，使用的是闵可夫斯基空间（l_p）。
#metric（矩阵）,默认为‘minkowski’。用于树的距离矩阵。默认为闵可夫斯基空间，如果和p=2一块使用相当于使用标准欧几里得矩阵。
#metric_params（矩阵参数）,默认为None

#4.模型评估
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#5.可视化
classes = list(set(Y_test))#set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
classes.sort()#sort函数对列表元素进行排序
plt.imshow(cm, cmap=plt.cm.Blues)#imshow热图，cmap颜色图谱
indices = range(len(cm))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()#添加颜色条形
plt.xlabel('predict')#x轴标签
plt.ylabel('real')#y轴标签
for first_index in range(len(cm)):
    for second_index in range(len(cm[first_index])):
        plt.text(first_index, second_index, cm[first_index][second_index])
plt.show()
#plt.text(x, y, s, fontsize, verticalalignment,horizontalalignment,rotation,kwargs)
#x,y为标签添加的位置；s：标签的符号，字符串格式
#fontsize标签字体大小；verticalalignment：垂直对齐方式 ，可选 ‘center’ ，‘top’ ， ‘bottom’，‘baseline’ 等
#horizontalalignment：水平对齐方式 ，可以填 ‘center’ ， ‘right’ ，‘left’ 等
#rotation：标签的旋转角度，以逆时针计算，取整

print(cm[first_index][second_index])#输出结果为29