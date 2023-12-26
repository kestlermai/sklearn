# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 20:54:04 2022

@author: 11146
"""

#Xgboost是“每一次的计算是都为了减少上一次的残差”的GBDT升级版。是一种把速度和效率发挥到极致的GBDT，所以叫作eXtreme Gradient Boosting。
#区别：（1）Xgboost可以使用正则项来控制模型的复杂度，防止过拟合；（2）Xgboost可以使用多种类型的基础分类器；
#（3）Xgboost在每轮迭代时，支持对数据进行随机采样（类似RF）；（4）Xgboost支持缺失值处理。

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

#自行安装xgboost，https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost 安装到Anaconda3\Scripts目录下，然后在该目录下的控制台
#pip install xgboost-1.6.1-cp39-cp39-win_amd64.whl 记得对应python版本选择下载

#构建xgboost模型
import xgboost as xgb
boost = xgb.XGBClassifier(n_estimators=200, max_depth=9, min_child_weight=3, 
                          subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1, gamma=0.1, reg_alpha=7)
boost.fit(X_train, Y_train)
#xgboost分为通用参数、booster参数、学习目标参数：用于控制训练目标的表现
#Booster参数，用于控制每一步的booster(tree/regression)，调参很大程度上都是在调整booster参数
#booster：指定要使用的基础模型，默认gbtree。可以输入gbtree，gblinear或dart。输入的评估器不同，使用的参数也不同，每种评估器都有自己的参数列表。评估器必须与自身的参数相匹配，否则报错。
#gbtree：即是论文中主要讨论的树模型，推荐使用；
#gblinear：是线性模型，表现很差，接近一个LASSO；
#dart：抛弃提升树，在建树的过程中会抛弃一部分树，比梯度提升树有更好的防过拟合功能。

#silent：是否为静默模式，默认0。都不推荐使用
#verbosity：打印模型构建过程信息的详细程度，默认1。可选值为0（静默）、1（警告）、2（信息）、3（调试）。

#booster参数：n_estimator：表示集成的基础模型的个数。跟随机森林的一样。
#learning_rate：每一步迭代的步长，也叫学习率，默认0.3。
#gamma：指定了节点分裂所需的最小损失函数下降值，值越大，算法越保守，默认为0。
#subsample：每棵树随机采样的比例。减小这个值，算法会更加保守，避免过拟合，默认为1。
#colsample_bytree：控制构建每棵树时随机抽样的特征占所有特征的比例，默认为1。
#colsample_bylevel：控制树在每一层分支时随机抽样出的特征占所有特征的比例，默认为1。
#max_depth：树的最大深度，用来控制过拟合，默认为6。
#max_delta_step：树的权重估计中允许的单次最大增量，默认为0。
#lambda：L2正则化项，默认为0。
#alpha：L1正则化项，默认为0。
#scale_pos_weight：在样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。通常可以将其设置为负样本的数目与正样本数目的比值。
#min_child_weight：定义一个子集的所有观察值的最小权重之和，同样是为了减少过拟合的参数，但是也不宜调得过高。

#学习目标参数
#objective：默认reg:linear。
#reg:linear – 线性回归；
#reg:logistic – 逻辑回归；
#binary:logistic – 二分类逻辑回归，输出为概率；
#binary:logitraw – 二分类逻辑回归，输出的结果为wTx；
#count:poisson – 计数问题的poisson回归，输出结果为poisson分布。
#multi:softmax – 设置 XGBoost 使用softmax目标函数做多分类，需要设置参数 num_class（类别个数）。
#multi:softprob – 如同softmax，但是输出结果为ndata*nclass的向量，其中的值是每个数据分为每个类的概率。

#eval_metric：默认为通过目标函数选择。
#rmse: 均方根误差；
#mae: 平均绝对值误差；
#logloss: negative log-likelihood
#error: 二分类错误率。其值通过错误分类数目与全部分类数目比值得到。对于预测，预测值大于0.5被认为是正类，其它归为负类。
#merror: 多分类错误率，计算公式为(wrong cases)/(all cases)。
#mlogloss: 多分类log损失
#auc: 曲线下的面积
#ndcg: Normalized Discounted Cumulative Gain，是用来衡量排序质量的指标
#map: 平均正确率

#预测结果
Y_pred = boost.predict(X_test)

#评估模型
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