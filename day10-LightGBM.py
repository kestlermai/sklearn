# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:33:04 2022

@author: 11146
"""

#LightGBM相对于xgboost，具有训练速度快和内存占用率低的特点

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

#构建lightgbm模型
import lightgbm as lgb
gbm = lgb.LGBMClassifier(boosting='gbdt', objective='binary', metric='auc', 
                         feature_fraction = 1, bagging_fraction = 0.9, lambda_l1 = 0.1, lambda_l2 = 0.1)
gbm.fit(X_train, Y_train)
#具体可以看：https://lightgbm.apachecn.org/#/docs/6
#boosting：指定要使用的基础模型，默认gbdt。可选项：gbdt，传统的梯度提升决策树；rf；dart，Dropouts meet Multiple Additive Regression Trees；goss, Gradient-based One-Side Sampling （基于梯度的单侧采样）。
#task：指定数据的用途，默认train。可选项：train，用于训练；predict，用于预测；convert_model，要将模型文件转换成 if-else 格式。
#objective：指定模型的用途，默认为regression。可选项：
#回归：regression，损失函数为L2；regression_l1，损失函数为L1；huber，损失函数为huber；fair，损失函数为fair；poisson，泊松回归；quantile，quantile 回归；quantile_l2, 类似于 quantile, 但是使用了L2损失函数。
#binary：二分类，用的最多的。
#多分类：multiclass，目标函数为softmax，需要指定num_class（分成几类）；multiclassova，目标函数One-vs-All，需要指定num_class（分成几类）。
#num_iterations：boosting的迭代次数，默认100。
#learning_rate：学习速度，默认0.1。
#num_leaves：一棵树上的叶子数，默认31。

#控制学习过程参数
#max_depth：指定树模型的最大深度，默认-1。
#min_data_in_leaf：一个叶子上数据的最小数量，默认20。
#feature_fraction：默认1.0。如果feature_fraction小于1.0，LightGBM将会在每次迭代中随机选择部分特征。例如，如果设置为0.8，将会在每棵树训练之前选择80%的特征。boosting 为random forest时用。
#feature_fraction_seed：随机选择特征时的随机种子数，默认2。
#bagging_fraction：每次迭代时用的数据比例，默认1。注意：为了启用bagging，bagging_freq应该设置为非零值。
#bagging_freq：bagging的频率，默认为0。设置为0意味着禁用bagging。k意味着每k次迭代执行bagging。
#bagging_seed：bagging 随机数种子，默认3。
#early_stopping_round：早停设置，默认为0。如果一次验证数据的一个度量在最近的early_stopping_round 回合中没有提高，模型将停止训练。
#lambda_l1：L1正则化，默认0。
#lambda_l2：L2正则化，默认0。
#min_gain_to_split：执行切分的最小增益，默认0。
#min_data_per_group：每个分类组的最小数据量，默认100。

#IO参数
#max_bin：表示特征将存入的 bin 的最大数量，默认255。
#categorical_feature：指定分类特征。用数字做索引，categorical_feature=0,1,2意味着column_0，column_1和column_2是分类特征。
#ignore_column：在培训中指定一些忽略的列。用数字做索引，ignore_column=0,1,2意味着column_0，column_1和column_2将被忽略。
#save_binary：默认Flase。设置为True时，则数据集被保存为二进制文件，下次读数据时速度会变快。

#学习目标参数
#is_unbalance：默认Flase。用于binary分类，如果数据不平衡设置为True。
#metric：设置度量指标：mae，mse，rmse，quantile，huber，fair，poisson，ndcg，map，auc，binary_logloss，binary_error（样本0的正确分类，1的错误分类），multi_logloss（mulit-class 损失日志分类），multi_error（error rate for mulit-class 出错率分类）。注意：支持多指标, 使用 , 分隔。

#预测结果
Y_pred = gbm.predict(X_test)

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