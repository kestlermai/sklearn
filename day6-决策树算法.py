# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:51:01 2022

@author: 11146
"""

#决策树（Decision Tree，DT），同样也是一个非常适合入门的分类算法，因为它的原理和KNN算法一样，很好理解和解释。
#DT对于数据量要求不是太高，且输入数据可以不需要进行规范化或者归一化处理。
#基本思想：通过在给定集合中使用自上而下的贪婪搜索算法来构造决策树，以测试每个树节点处的每个属性
#循环A——很好的属性，然后A分配每个节点的决策属性，对A每个值分别创建一个字节点，如果能很好地进行分类，那么停止循环，否则继续创建节点

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier#决策树分类算法
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 导入数据集
dataset = pd.read_csv('Day 4 Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
Y = dataset.iloc[:, 4].values

Labelencoder = LabelEncoder()
X[:, 0] = Labelencoder.fit_transform(X[:, 0])# 性别转化为数字


# 将数据集分成训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#构建决策树模型
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)#节点质量评估函数选用熵
classifier.fit(X_train, Y_train)
#决策树代码
#clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, 
                #min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, 
                #random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.,  class_weight=None, ccp_alpha=0.0)
#criterion标准，gini（默认），entropy。节点质量评估函数（gini为基尼系数，entropy为熵）。
#splitter分束器，best（默认），random。分枝时变量选择方式（random：随机选择，best：选择最好的变量）。
#max_depth：整数，默认None。树分枝的最大深度（为None时，树分枝深度无限制）
#min_samples_split：整数或小数，默认2。节点分枝最小样本个数。节点样本>=min_samples_split时，允许分枝，如果小于该值，则不再分枝（也可以设为小数，此时当作总样本占比，即min_samples_split=ceil(min_samples_split *总样本数）。
#min_samples_leaf：整数或小数，默认1。叶子节点最小样本数。左右节点都需要满足>=min_samples_leaf,才会将父节点分枝，如果小于该值，则不再分枝（也可以设为小数，此时当作总样本占比，即min_samples_split=ceil(min_samples_split *总样本数)）。
#min_weight_fraction_leaf：小数，默认值0。叶子节点最小权重和。节点作为叶子节点，样本权重总和必须>=min_weight_fraction_leaf,为0时即无限制。
#max_features：整数，小数，None(默认)，{"auto", "sqrt", "log2"}。特征最大查找个数。先对max_features进行如下转换，统一转换成成整数。
#整数：max_features=max_features
#auto：max_features=sqrt(n_features)
#sqrt：max_features=sqrt(n_features)
#log2：max_features=log2rt(n_features)
#小数：max_features=int(max_features * n_features)
#None:max_features=n_features
#如果max_features<特征个数，则会随机抽取max_features个特征，只在这max_features中查找特征进行分裂。
#random_state：整数，None(默认)。需要每次训练都一样时，就需要设置该参数。

#max_leaf_nodes：整数，None（默认）。最大叶子节点数。如果为None,则无限制。
#min_impurity_decrease：小数，默认0。节点分枝最小纯度增长量。信息增益
#class_weight：设置各类别样本的权重，默认是各个样本权重一样，都为1。
#ccp_alpha：剪枝时的alpha系数，需要剪枝时设置该参数，默认值是不会剪枝的。

#评估模型
Y_pred = classifier.predict(X_test)
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