# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:43:48 2022

@author: 11146
"""
#ps：在predict之前模型都需要fit
#集成学习的原理
#生成多个单一的学习器，然后把所有学习器汇总，通过某种策略组成强学习器
#单一学习器：logistics回归、KNN、决策树等等
#RF属于经典的Bagging模型，Bagging对待所有基础模型一视同仁
#AdaBoost模型属于经典Boosting模型，做到“区别对待”，注重“培养精英”（优秀模型权重大）和“重视错误”（减少预测错误的权重）
#Boosting（提升算法）：训练过程技术路线呈阶梯状，逐一进行训练，每一轮训练的训练集均不同，假设某一个数据本轮分错了，那么在下轮就会分配它更大的权重。
#梯度提升树（Gradient Boosting Decision Tree，GBDT），不同于Adaboost利用前一轮训练的误差来更新下一轮学习的样本权重，GBDT每次都拟合上一轮模型产生的误差。
#Xgboost（eXtreme Gradient Boosting），它是GBDT的改进版本，公认的大杀器

#Bagging（训练多个分类器取平均）：从训练集有放回地随机抽样生成一系列子训练集，分别训练一些列单一模型，将所有单一模型的输出结果采用投票的方式得到最终结果。
#Stacking（堆叠各种分类器）
#第一个阶段的模型是以原始训练集为输入，况且叫做L1模型。
#第二个阶段的模型是以L1模型在原始训练集上的拟合值作为训练集，以L1模型在原始测试集上的预测作为测试集，也就是最终模型了。

#AdaBoost、GBDT、Stacking实现
import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径
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

#构建AdaBoostClassifier模型
from sklearn.ensemble import AdaBoostClassifier
classifier_ada = AdaBoostClassifier(n_estimators=100)
classifier_ada = classifier_ada.fit(X_train,Y_train)
#class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, 
                                            #learning_rate=1.0, algorithm='SAMME.R', random_state=None)
#base_estimator：object, 默认None。建立增强集成的基础估计器。需要支持示例权重，以及适当的classes_和n_classes_属性。
#如果没有，那么基础估计器是DecisionTreeClassifier(max_depth=1)。
#n_estimators：整数，默认50。终止推进的估计器的最大数目。如果完全拟合，学习过程就会提前停止。
#learning_rate：整数，默认1。学习率通过learning_rate缩小每个分类器的贡献程度。learning_rate和n_estimators之间存在权衡关系。
#algorithm：{'SAMME', 'SAMME.R'}，默认'SAMME.R'。若为"SAMME.R"则使用real bossting算法。
#base_estimator必须支持类概率的计算。若为SAMME，则使用discrete boosting算法。SAMME.R算法的收敛速度通常比SAMME快，通过更少的增强迭代获得更低的测试误差。
#random_state：小数，默认None。控制每个base_estimator在每个增强迭代中给定的随机种子。
#因此，仅在base_estimator引入random_state时使用它。在多个函数调用之间传递可重复输出的整数。

#评估AdaBoostClassifier模型
Y_pred = classifier_ada.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#可视化
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

#构建GBDT模型
from sklearn.ensemble import GradientBoostingClassifier
classifier_gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
classifier_gbdt.fit(X_train, Y_train)
#class sklearn.ensemble.GradientBoostingClassifier(*, loss='deviance', learning_rate=0.1, 
            #n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
            #min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
            #min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, 
            #max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, 
            #n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
#后续使用xgboost，GBDT参数自行上官网查看https://scikit-learn.org.cn/view/628.html

#评估GBDT模型
Y_pred = classifier_ada.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#可视化
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

#构建Stacking堆积多种分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=0)),
           ('knn', KNeighborsClassifier(n_neighbors=5)),
           ('dt', DecisionTreeClassifier(criterion = 'entropy', random_state = 0))]
classifier_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
classifier_stack.fit(X_train, Y_train)

#class sklearn.ensemble.StackingClassifier(estimators, final_estimator=None, *, cv=None, 
                                            #stack_method='auto', n_jobs=None, passthrough=False, verbose=0)
#estimators：基础模型的列表。将被堆叠在一起的基础估计器。列表中的每个元素都被定义为一个字符串元组(即名称)和一个estimator实例。
#可以使用set_params将评估器设置为“drop”。

#final_estimator：一个分类器，它将被用来组合基础估计器。默认的分类器是LogisticRegression。
#cv：交叉认证策略。None，使用默认的5折交叉验证；整数，用于指定(分层的)K-Fold中的折叠数。

#stack_method：{‘auto’, ‘predict_proba’, ‘decision_function’, ‘predict’}, 默认’auto’。
#每个基本估计器调用的方法。如果是“auto”，它将对每个估计器按相应顺序调用“predict_proba”、“decision_function”或“predict”。
#否则，调用'predict_proba'、'decision_function'或'predict'中的一个。如果估计器没有实现该方法，它将产生一个错误。

#passthrough：布尔值, 默认False。当为False时，只使用估计器的预测作为final_estimator的训练数据。
#当为真时，final_estimator将在预测和原始训练数据上进行训练。

#Stacking模型评估
Y_pred = classifier_stack.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#可视化
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