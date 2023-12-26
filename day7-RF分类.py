# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:29:06 2022

@author: 11146
"""
#随机森林（randomForest）通过构建许多决策树进行集成学习，属于有监督集成学习模型，即可分类又可回归
#经典的Bagging模型，Bagging对待所有基础模型一视同仁，而Boosting做到“区别对待”，注重“培养精英”（优秀模型权重大）和“重视错误”（减少预测错误的权重）
#首先是随机选择数据。进行有放回的随机采样，比如说随机抽取70%的数据当作第一棵树的输入，再随机抽取70%的数据当作第二棵树的输入，这样构建的树有自己的个性。
#其次是随机选择特征。第一棵树随机选择所有特征中的60%来建模，第二棵再随机选择其中60%的特征来建模，这样就把差异放大了。
#第一步构建决策树，第二步根据决策树的分类器结果做出预测；跟决策树的最大区别是根节点和分割特征节点的过程是随机进行的

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier#RF分类算法
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

#构建RF模型
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

#class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', 
            #max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
            #max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
            #oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, 
            #class_weight=None, ccp_alpha=0.0, max_samples=None)
#n_estimators：整数，默认10。决策树的个数，理论上越多越好，注意理论上，还是得根据数据集调整的。           
#bootstrap：默认Ture。是否有放回的采样
#oob_score：默认False。袋外数据，也就是在某次随机选取一批数据来构建决策树，那些没有被选取到的数据。可以用来做一个简单的交叉验证，性能消耗小，效果还行。


Y_pred = classifier.predict(X_test)
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

#思考
#这4+4=8个样本到底长什么样子，使得模型都分不开呢？
#森林中任意两棵树的相关性：相关性越大，错误率越大；
#森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。
#减小特征选择个数m，树的相关性和分类能力也会相应的降低；
#增大m，两者也会随之增大。所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。